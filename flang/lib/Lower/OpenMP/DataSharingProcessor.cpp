//===-- DataSharingProcessor.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "DataSharingProcessor.h"

#include "Utils.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Semantics/tools.h"

namespace Fortran {
namespace lower {
namespace omp {

void DataSharingProcessor::processStep1(
    mlir::omp::PrivateClauseOps *clauseOps,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms) {
  collectSymbolsForPrivatization();
  collectDefaultSymbols();
  privatize(clauseOps, privateSyms);
  defaultPrivatize(clauseOps, privateSyms);
  insertBarrier();
}

void DataSharingProcessor::processStep2(mlir::Operation *op, bool isLoop) {
  insPt = firOpBuilder.saveInsertionPoint();
  copyLastPrivatize(op);
  firOpBuilder.restoreInsertionPoint(insPt);

  if (isLoop) {
    // push deallocs out of the loop
    firOpBuilder.setInsertionPointAfter(op);
    insertDeallocs();
  } else {
    // insert dummy instruction to mark the insertion position
    mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
        op->getLoc(), firOpBuilder.getIndexType());
    insertDeallocs();
    firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  }
}

void DataSharingProcessor::insertDeallocs() {
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols)
    if (Fortran::semantics::IsAllocatable(sym->GetUltimate())) {
      if (!useDelayedPrivatization) {
        converter.createHostAssociateVarCloneDealloc(*sym);
        return;
      }

      Fortran::lower::SymbolBox hsb = converter.lookupOneLevelUpSymbol(*sym);
      assert(hsb && "Host symbol box not found");
      mlir::Type symType = hsb.getAddr().getType();
      mlir::Location symLoc = hsb.getAddr().getLoc();
      fir::ExtendedValue symExV = converter.getSymbolExtendedValue(*sym);
      mlir::omp::PrivateClauseOp privatizer = symToPrivatizer.at(sym);

      symTable->pushScope();

      mlir::OpBuilder::InsertionGuard guard(firOpBuilder);

      mlir::Region &deallocRegion = privatizer.getDeallocRegion();
      fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
      mlir::Block *deallocEntryBlock = firOpBuilder.createBlock(
          &deallocRegion, /*insertPt=*/{}, symType, symLoc);

      firOpBuilder.setInsertionPointToEnd(deallocEntryBlock);
      symTable->addSymbol(*sym,
                          fir::substBase(symExV, deallocRegion.getArgument(0)));

      converter.createHostAssociateVarCloneDealloc(*sym);
      firOpBuilder.create<mlir::omp::YieldOp>(hsb.getAddr().getLoc());

      symTable->popScope();
    }
}

void DataSharingProcessor::cloneSymbol(const Fortran::semantics::Symbol *sym) {
  // Privatization for symbols which are pre-determined (like loop index
  // variables) happen separately, for everything else privatize here.
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpPreDetermined))
    return;
  bool success = converter.createHostAssociateVarClone(*sym);
  (void)success;
  assert(success && "Privatization failed due to existing binding");
}

void DataSharingProcessor::copyFirstPrivateSymbol(
    const Fortran::semantics::Symbol *sym,
    mlir::OpBuilder::InsertPoint *copyAssignIP) {
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate))
    converter.copyHostAssociateVar(*sym, copyAssignIP);
}

void DataSharingProcessor::copyLastPrivateSymbol(
    const Fortran::semantics::Symbol *sym,
    [[maybe_unused]] mlir::OpBuilder::InsertPoint *lastPrivIP) {
  if (sym->test(Fortran::semantics::Symbol::Flag::OmpLastPrivate))
    converter.copyHostAssociateVar(*sym, lastPrivIP);
}

void DataSharingProcessor::collectOmpObjectListSymbol(
    const omp::ObjectList &objects,
    llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet) {
  for (const omp::Object &object : objects)
    symbolSet.insert(object.id());
}

void DataSharingProcessor::collectSymbolsForPrivatization() {
  bool hasCollapse = false;
  for (const omp::Clause &clause : clauses) {
    if (const auto &privateClause =
            std::get_if<omp::clause::Private>(&clause.u)) {
      collectOmpObjectListSymbol(privateClause->v, privatizedSymbols);
    } else if (const auto &firstPrivateClause =
                   std::get_if<omp::clause::Firstprivate>(&clause.u)) {
      collectOmpObjectListSymbol(firstPrivateClause->v, privatizedSymbols);
    } else if (const auto &lastPrivateClause =
                   std::get_if<omp::clause::Lastprivate>(&clause.u)) {
      const ObjectList &objects = std::get<ObjectList>(lastPrivateClause->t);
      collectOmpObjectListSymbol(objects, privatizedSymbols);
      hasLastPrivateOp = true;
    } else if (std::get_if<omp::clause::Collapse>(&clause.u)) {
      hasCollapse = true;
    }
  }

  if (hasCollapse && hasLastPrivateOp)
    TODO(converter.getCurrentLocation(), "Collapse clause with lastprivate");
}

bool DataSharingProcessor::needBarrier() {
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols) {
    if (sym->test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate) &&
        sym->test(Fortran::semantics::Symbol::Flag::OmpLastPrivate))
      return true;
  }
  return false;
}

void DataSharingProcessor::insertBarrier() {
  // Emit implicit barrier to synchronize threads and avoid data races on
  // initialization of firstprivate variables and post-update of lastprivate
  // variables.
  // FIXME: Emit barrier for lastprivate clause when 'sections' directive has
  // 'nowait' clause. Otherwise, emit barrier when 'sections' directive has
  // both firstprivate and lastprivate clause.
  // Emit implicit barrier for linear clause. Maybe on somewhere else.
  if (needBarrier())
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
}

void DataSharingProcessor::insertLastPrivateCompare(mlir::Operation *op) {
  mlir::omp::LoopNestOp loopOp;
  if (auto wrapper = mlir::dyn_cast<mlir::omp::LoopWrapperInterface>(op))
    loopOp = wrapper.isWrapper()
                 ? mlir::cast<mlir::omp::LoopNestOp>(wrapper.getWrappedLoop())
                 : nullptr;

  bool cmpCreated = false;
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  for (const omp::Clause &clause : clauses) {
    if (clause.id != llvm::omp::OMPC_lastprivate)
      continue;
    // TODO: Add lastprivate support for simd construct
    if (mlir::isa<mlir::omp::SectionOp>(op)) {
      if (&eval == &eval.parentConstruct->getLastNestedEvaluation()) {
        // For `omp.sections`, lastprivatized variables occur in
        // lexically final `omp.section` operation. The following FIR
        // shall be generated for the same:
        //
        // omp.sections lastprivate(...) {
        //  omp.section {...}
        //  omp.section {...}
        //  omp.section {
        //      fir.allocate for `private`/`firstprivate`
        //      <More operations here>
        //      fir.if %true {
        //          ^%lpv_update_blk
        //      }
        //  }
        // }
        //
        // To keep code consistency while handling privatization
        // through this control flow, add a `fir.if` operation
        // that always evaluates to true, in order to create
        // a dedicated sub-region in `omp.section` where
        // lastprivate FIR can reside. Later canonicalizations
        // will optimize away this operation.
        if (!eval.lowerAsUnstructured()) {
          auto ifOp = firOpBuilder.create<fir::IfOp>(
              op->getLoc(),
              firOpBuilder.createIntegerConstant(
                  op->getLoc(), firOpBuilder.getIntegerType(1), 0x1),
              /*else*/ false);
          firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());

          const Fortran::parser::OpenMPConstruct *parentOmpConstruct =
              eval.parentConstruct->getIf<Fortran::parser::OpenMPConstruct>();
          assert(parentOmpConstruct &&
                 "Expected a valid enclosing OpenMP construct");
          const Fortran::parser::OpenMPSectionsConstruct *sectionsConstruct =
              std::get_if<Fortran::parser::OpenMPSectionsConstruct>(
                  &parentOmpConstruct->u);
          assert(sectionsConstruct &&
                 "Expected an enclosing omp.sections construct");
          const Fortran::parser::OmpClauseList &sectionsEndClauseList =
              std::get<Fortran::parser::OmpClauseList>(
                  std::get<Fortran::parser::OmpEndSectionsDirective>(
                      sectionsConstruct->t)
                      .t);
          for (const Fortran::parser::OmpClause &otherClause :
               sectionsEndClauseList.v)
            if (std::get_if<Fortran::parser::OmpClause::Nowait>(&otherClause.u))
              // Emit implicit barrier to synchronize threads and avoid data
              // races on post-update of lastprivate variables when `nowait`
              // clause is present.
              firOpBuilder.create<mlir::omp::BarrierOp>(
                  converter.getCurrentLocation());
          firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          lastPrivIP = firOpBuilder.saveInsertionPoint();
          firOpBuilder.setInsertionPoint(ifOp);
          insPt = firOpBuilder.saveInsertionPoint();
        } else {
          // Lastprivate operation is inserted at the end
          // of the lexically last section in the sections
          // construct
          mlir::OpBuilder::InsertionGuard unstructuredSectionsGuard(
              firOpBuilder);
          mlir::Operation *lastOper = op->getRegion(0).back().getTerminator();
          firOpBuilder.setInsertionPoint(lastOper);
          lastPrivIP = firOpBuilder.saveInsertionPoint();
        }
      }
    } else if (mlir::isa<mlir::omp::WsloopOp>(op)) {
      // Update the original variable just before exiting the worksharing
      // loop. Conversion as follows:
      //
      // omp.wsloop {             omp.wsloop {
      //   omp.loop_nest {          omp.loop_nest {
      //     ...                      ...
      //     store          ===>      store
      //     omp.yield                %v = arith.addi %iv, %step
      //   }                          %cmp = %step < 0 ? %v < %ub : %v > %ub
      //   omp.terminator             fir.if %cmp {
      // }                              fir.store %v to %loopIV
      //                                ^%lpv_update_blk:
      //                              }
      //                              omp.yield
      //                            }
      //                            omp.terminator
      //                          }

      // Only generate the compare once in presence of multiple LastPrivate
      // clauses.
      if (cmpCreated)
        continue;
      cmpCreated = true;

      mlir::Location loc = loopOp.getLoc();
      mlir::Operation *lastOper = loopOp.getRegion().back().getTerminator();
      firOpBuilder.setInsertionPoint(lastOper);

      mlir::Value iv = loopOp.getIVs()[0];
      mlir::Value ub = loopOp.getUpperBound()[0];
      mlir::Value step = loopOp.getStep()[0];

      // v = iv + step
      // cmp = step < 0 ? v < ub : v > ub
      mlir::Value v = firOpBuilder.create<mlir::arith::AddIOp>(loc, iv, step);
      mlir::Value zero =
          firOpBuilder.createIntegerConstant(loc, step.getType(), 0);
      mlir::Value negativeStep = firOpBuilder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, step, zero);
      mlir::Value vLT = firOpBuilder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, v, ub);
      mlir::Value vGT = firOpBuilder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, v, ub);
      mlir::Value cmpOp = firOpBuilder.create<mlir::arith::SelectOp>(
          loc, negativeStep, vLT, vGT);

      auto ifOp = firOpBuilder.create<fir::IfOp>(loc, cmpOp, /*else*/ false);
      firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      assert(loopIV && "loopIV was not set");
      firOpBuilder.create<fir::StoreOp>(loopOp.getLoc(), v, loopIV);
      lastPrivIP = firOpBuilder.saveInsertionPoint();
    } else {
      TODO(converter.getCurrentLocation(),
           "lastprivate clause in constructs other than "
           "simd/worksharing-loop");
    }
  }
}

void DataSharingProcessor::collectSymbolsInNestedRegions(
    Fortran::lower::pft::Evaluation &eval,
    Fortran::semantics::Symbol::Flag flag,
    llvm::SetVector<const Fortran::semantics::Symbol *>
        &symbolsInNestedRegions) {
  for (Fortran::lower::pft::Evaluation &nestedEval :
       eval.getNestedEvaluations()) {
    if (nestedEval.hasNestedEvaluations()) {
      if (nestedEval.isConstruct())
        // Recursively look for OpenMP constructs within `nestedEval`'s region
        collectSymbolsInNestedRegions(nestedEval, flag, symbolsInNestedRegions);
      else
        converter.collectSymbolSet(nestedEval, symbolsInNestedRegions, flag,
                                   /*collectSymbols=*/true,
                                   /*collectHostAssociatedSymbols=*/false);
    }
  }
}

// Collect symbols to be default privatized in two steps.
// In step 1, collect all symbols in `eval` that match `flag` into
// `defaultSymbols`. In step 2, for nested constructs (if any), if and only if
// the nested construct is an OpenMP construct, collect those nested
// symbols skipping host associated symbols into `symbolsInNestedRegions`.
// Later, in current context, all symbols in the set
// `defaultSymbols` - `symbolsInNestedRegions` will be privatized.
void DataSharingProcessor::collectSymbols(
    Fortran::semantics::Symbol::Flag flag) {
  converter.collectSymbolSet(eval, defaultSymbols, flag,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  collectSymbolsInNestedRegions(eval, flag, symbolsInNestedRegions);
}

void DataSharingProcessor::collectDefaultSymbols() {
  using DataSharingAttribute = omp::clause::Default::DataSharingAttribute;
  for (const omp::Clause &clause : clauses) {
    if (const auto *defaultClause =
            std::get_if<omp::clause::Default>(&clause.u)) {
      if (defaultClause->v == DataSharingAttribute::Private)
        collectSymbols(Fortran::semantics::Symbol::Flag::OmpPrivate);
      else if (defaultClause->v == DataSharingAttribute::Firstprivate)
        collectSymbols(Fortran::semantics::Symbol::Flag::OmpFirstPrivate);
    }
  }
}

void DataSharingProcessor::privatize(
    mlir::omp::PrivateClauseOps *clauseOps,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms) {
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols) {
    if (const auto *commonDet =
            sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects())
        doPrivatize(&*mem, clauseOps, privateSyms);
    } else
      doPrivatize(sym, clauseOps, privateSyms);
  }
}

void DataSharingProcessor::copyLastPrivatize(mlir::Operation *op) {
  insertLastPrivateCompare(op);
  for (const Fortran::semantics::Symbol *sym : privatizedSymbols)
    if (const auto *commonDet =
            sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects()) {
        copyLastPrivateSymbol(&*mem, &lastPrivIP);
      }
    } else {
      copyLastPrivateSymbol(sym, &lastPrivIP);
    }
}

void DataSharingProcessor::defaultPrivatize(
    mlir::omp::PrivateClauseOps *clauseOps,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms) {
  for (const Fortran::semantics::Symbol *sym : defaultSymbols) {
    if (!Fortran::semantics::IsProcedure(*sym) &&
        !sym->GetUltimate().has<Fortran::semantics::DerivedTypeDetails>() &&
        !sym->GetUltimate().has<Fortran::semantics::NamelistDetails>() &&
        !Fortran::semantics::IsImpliedDoIndex(sym->GetUltimate()) &&
        !symbolsInNestedRegions.contains(sym) &&
        !privatizedSymbols.contains(sym))
      doPrivatize(sym, clauseOps, privateSyms);
  }
}

void DataSharingProcessor::doPrivatize(
    const Fortran::semantics::Symbol *sym,
    mlir::omp::PrivateClauseOps *clauseOps,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms) {
  if (!useDelayedPrivatization) {
    cloneSymbol(sym);
    copyFirstPrivateSymbol(sym);
    return;
  }

  Fortran::lower::SymbolBox hsb = converter.lookupOneLevelUpSymbol(*sym);
  assert(hsb && "Host symbol box not found");

  mlir::Type symType = hsb.getAddr().getType();
  mlir::Location symLoc = hsb.getAddr().getLoc();
  std::string privatizerName = sym->name().ToString() + ".privatizer";
  bool isFirstPrivate =
      sym->test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate);

  mlir::omp::PrivateClauseOp privatizerOp = [&]() {
    auto moduleOp = firOpBuilder.getModule();
    auto uniquePrivatizerName = fir::getTypeAsString(
        symType, converter.getKindMap(),
        converter.mangleName(*sym) +
            (isFirstPrivate ? "_firstprivate" : "_private"));

    if (auto existingPrivatizer =
            moduleOp.lookupSymbol<mlir::omp::PrivateClauseOp>(
                uniquePrivatizerName))
      return existingPrivatizer;

    mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
    firOpBuilder.setInsertionPoint(&moduleOp.getBodyRegion().front(),
                                   moduleOp.getBodyRegion().front().begin());
    auto result = firOpBuilder.create<mlir::omp::PrivateClauseOp>(
        symLoc, uniquePrivatizerName, symType,
        isFirstPrivate ? mlir::omp::DataSharingClauseType::FirstPrivate
                       : mlir::omp::DataSharingClauseType::Private);
    fir::ExtendedValue symExV = converter.getSymbolExtendedValue(*sym);

    symTable->pushScope();

    // Populate the `alloc` region.
    {
      mlir::Region &allocRegion = result.getAllocRegion();
      mlir::Block *allocEntryBlock = firOpBuilder.createBlock(
          &allocRegion, /*insertPt=*/{}, symType, symLoc);

      firOpBuilder.setInsertionPointToEnd(allocEntryBlock);
      symTable->addSymbol(*sym,
                          fir::substBase(symExV, allocRegion.getArgument(0)));
      symTable->pushScope();
      cloneSymbol(sym);
      firOpBuilder.create<mlir::omp::YieldOp>(
          hsb.getAddr().getLoc(),
          symTable->shallowLookupSymbol(*sym).getAddr());
      symTable->popScope();
    }

    // Populate the `copy` region if this is a `firstprivate`.
    if (isFirstPrivate) {
      mlir::Region &copyRegion = result.getCopyRegion();
      // First block argument corresponding to the original/host value while
      // second block argument corresponding to the privatized value.
      mlir::Block *copyEntryBlock = firOpBuilder.createBlock(
          &copyRegion, /*insertPt=*/{}, {symType, symType}, {symLoc, symLoc});
      firOpBuilder.setInsertionPointToEnd(copyEntryBlock);
      symTable->addSymbol(*sym,
                          fir::substBase(symExV, copyRegion.getArgument(0)),
                          /*force=*/true);
      symTable->pushScope();
      symTable->addSymbol(*sym,
                          fir::substBase(symExV, copyRegion.getArgument(1)));
      auto ip = firOpBuilder.saveInsertionPoint();
      copyFirstPrivateSymbol(sym, &ip);

      firOpBuilder.create<mlir::omp::YieldOp>(
          hsb.getAddr().getLoc(),
          symTable->shallowLookupSymbol(*sym).getAddr());
      symTable->popScope();
    }

    symTable->popScope();
    return result;
  }();

  if (clauseOps) {
    clauseOps->privatizers.push_back(mlir::SymbolRefAttr::get(privatizerOp));
    clauseOps->privateVars.push_back(hsb.getAddr());
  }

  if (privateSyms)
    privateSyms->push_back(sym);

  symToPrivatizer[sym] = privatizerOp;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
