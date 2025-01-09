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
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Semantics/tools.h"

namespace Fortran {
namespace lower {
namespace omp {
bool DataSharingProcessor::OMPConstructSymbolVisitor::isSymbolDefineBy(
    const semantics::Symbol *symbol, lower::pft::Evaluation &eval) const {
  return eval.visit(
      common::visitors{[&](const parser::OpenMPConstruct &functionParserNode) {
                         return symDefMap.count(symbol) &&
                                symDefMap.at(symbol) == &functionParserNode;
                       },
                       [](const auto &functionParserNode) { return false; }});
}

DataSharingProcessor::DataSharingProcessor(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, lower::pft::Evaluation &eval,
    bool shouldCollectPreDeterminedSymbols, bool useDelayedPrivatization,
    lower::SymMap &symTable)
    : converter(converter), semaCtx(semaCtx),
      firOpBuilder(converter.getFirOpBuilder()), clauses(clauses), eval(eval),
      shouldCollectPreDeterminedSymbols(shouldCollectPreDeterminedSymbols),
      useDelayedPrivatization(useDelayedPrivatization), symTable(symTable),
      visitor() {
  eval.visit([&](const auto &functionParserNode) {
    parser::Walk(functionParserNode, visitor);
  });
}

void DataSharingProcessor::processStep1(
    mlir::omp::PrivateClauseOps *clauseOps) {
  collectSymbolsForPrivatization();
  collectDefaultSymbols();
  collectImplicitSymbols();
  collectPreDeterminedSymbols();

  privatize(clauseOps);

  insertBarrier();
}

void DataSharingProcessor::processStep2(mlir::Operation *op, bool isLoop) {
  // 'sections' lastprivate is handled by genOMP()
  if (!mlir::isa<mlir::omp::SectionsOp>(op)) {
    mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
    copyLastPrivatize(op);
  }

  if (isLoop) {
    // push deallocs out of the loop
    firOpBuilder.setInsertionPointAfter(op);
    insertDeallocs();
  } else {
    mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
    insertDeallocs();
  }
}

void DataSharingProcessor::insertDeallocs() {
  for (const semantics::Symbol *sym : allPrivatizedSymbols)
    if (semantics::IsAllocatable(sym->GetUltimate())) {
      if (!useDelayedPrivatization) {
        converter.createHostAssociateVarCloneDealloc(*sym);
        continue;
      }

      lower::SymbolBox hsb = converter.lookupOneLevelUpSymbol(*sym);
      assert(hsb && "Host symbol box not found");
      mlir::Type symType = hsb.getAddr().getType();
      mlir::Location symLoc = hsb.getAddr().getLoc();
      fir::ExtendedValue symExV = converter.getSymbolExtendedValue(*sym);
      mlir::omp::PrivateClauseOp privatizer = symToPrivatizer.at(sym);

      lower::SymMapScope scope(symTable);
      mlir::OpBuilder::InsertionGuard guard(firOpBuilder);

      mlir::Region &deallocRegion = privatizer.getDeallocRegion();
      fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
      mlir::Block *deallocEntryBlock = firOpBuilder.createBlock(
          &deallocRegion, /*insertPt=*/{}, symType, symLoc);

      firOpBuilder.setInsertionPointToEnd(deallocEntryBlock);
      symTable.addSymbol(*sym,
                         fir::substBase(symExV, deallocRegion.getArgument(0)));

      converter.createHostAssociateVarCloneDealloc(*sym);
      firOpBuilder.create<mlir::omp::YieldOp>(hsb.getAddr().getLoc());
    }
}

void DataSharingProcessor::cloneSymbol(const semantics::Symbol *sym) {
  bool isFirstPrivate = sym->test(semantics::Symbol::Flag::OmpFirstPrivate);
  bool success = converter.createHostAssociateVarClone(
      *sym, /*skipDefaultInit=*/isFirstPrivate);
  (void)success;
  assert(success && "Privatization failed due to existing binding");

  // Initialize clone from original object if it has any allocatable member.
  auto needInitClone = [&] {
    if (isFirstPrivate)
      return false;

    SymbolBox sb = symTable.lookupSymbol(sym);
    assert(sb);
    mlir::Value addr = sb.getAddr();
    assert(addr);
    return !fir::isPointerType(addr.getType()) &&
           hlfir::mayHaveAllocatableComponent(addr.getType());
  };

  if (needInitClone()) {
    Fortran::lower::initializeCloneAtRuntime(converter, *sym, symTable);
    callsInitClone = true;
  }
}

void DataSharingProcessor::copyFirstPrivateSymbol(
    const semantics::Symbol *sym, mlir::OpBuilder::InsertPoint *copyAssignIP) {
  if (sym->test(semantics::Symbol::Flag::OmpFirstPrivate))
    converter.copyHostAssociateVar(*sym, copyAssignIP);
}

void DataSharingProcessor::copyLastPrivateSymbol(
    const semantics::Symbol *sym, mlir::OpBuilder::InsertPoint *lastPrivIP) {
  if (sym->test(semantics::Symbol::Flag::OmpLastPrivate))
    converter.copyHostAssociateVar(*sym, lastPrivIP);
}

void DataSharingProcessor::collectOmpObjectListSymbol(
    const omp::ObjectList &objects,
    llvm::SetVector<const semantics::Symbol *> &symbolSet) {
  for (const omp::Object &object : objects)
    symbolSet.insert(object.sym());
}

void DataSharingProcessor::collectSymbolsForPrivatization() {
  for (const omp::Clause &clause : clauses) {
    if (const auto &privateClause =
            std::get_if<omp::clause::Private>(&clause.u)) {
      collectOmpObjectListSymbol(privateClause->v, explicitlyPrivatizedSymbols);
    } else if (const auto &firstPrivateClause =
                   std::get_if<omp::clause::Firstprivate>(&clause.u)) {
      collectOmpObjectListSymbol(firstPrivateClause->v,
                                 explicitlyPrivatizedSymbols);
    } else if (const auto &lastPrivateClause =
                   std::get_if<omp::clause::Lastprivate>(&clause.u)) {
      lastprivateModifierNotSupported(*lastPrivateClause,
                                      converter.getCurrentLocation());
      const ObjectList &objects = std::get<ObjectList>(lastPrivateClause->t);
      collectOmpObjectListSymbol(objects, explicitlyPrivatizedSymbols);
    }
  }

  for (auto *sym : explicitlyPrivatizedSymbols)
    allPrivatizedSymbols.insert(sym);
}

bool DataSharingProcessor::needBarrier() {
  // Emit implicit barrier to synchronize threads and avoid data races on
  // initialization of firstprivate variables and post-update of lastprivate
  // variables.
  // Emit implicit barrier for linear clause. Maybe on somewhere else.
  for (const semantics::Symbol *sym : allPrivatizedSymbols) {
    if (sym->test(semantics::Symbol::Flag::OmpLastPrivate) &&
        (sym->test(semantics::Symbol::Flag::OmpFirstPrivate) || callsInitClone))
      return true;
  }
  return false;
}

void DataSharingProcessor::insertBarrier() {
  if (needBarrier())
    firOpBuilder.create<mlir::omp::BarrierOp>(converter.getCurrentLocation());
}

void DataSharingProcessor::insertLastPrivateCompare(mlir::Operation *op) {
  mlir::omp::LoopNestOp loopOp;
  if (auto wrapper = mlir::dyn_cast<mlir::omp::LoopWrapperInterface>(op))
    loopOp = mlir::cast<mlir::omp::LoopNestOp>(wrapper.getWrappedLoop());

  bool cmpCreated = false;
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  for (const omp::Clause &clause : clauses) {
    if (clause.id != llvm::omp::OMPC_lastprivate)
      continue;
    if (mlir::isa<mlir::omp::WsloopOp>(op) ||
        mlir::isa<mlir::omp::SimdOp>(op)) {
      // Update the original variable just before exiting the worksharing
      // loop. Conversion as follows:
      //
      // omp.wsloop / omp.simd {    omp.wsloop / omp.simd {
      //   omp.loop_nest {            omp.loop_nest {
      //     ...                        ...
      //     store          ===>        store
      //     omp.yield                  %v = arith.addi %iv, %step
      //   }                            %cmp = %step < 0 ? %v < %ub : %v > %ub
      // }                              fir.if %cmp {
      //                                  fir.store %v to %loopIV
      //                                  ^%lpv_update_blk:
      //                                }
      //                                omp.yield
      //                              }
      //                            }

      // Only generate the compare once in presence of multiple LastPrivate
      // clauses.
      if (cmpCreated)
        continue;
      cmpCreated = true;

      mlir::Location loc = loopOp.getLoc();
      mlir::Operation *lastOper = loopOp.getRegion().back().getTerminator();
      firOpBuilder.setInsertionPoint(lastOper);

      mlir::Value cmpOp;
      llvm::SmallVector<mlir::Value> vs;
      vs.reserve(loopOp.getIVs().size());
      for (auto [iv, ub, step] :
           llvm::zip_equal(loopOp.getIVs(), loopOp.getLoopUpperBounds(),
                           loopOp.getLoopSteps())) {
        // v = iv + step
        // cmp = step < 0 ? v < ub : v > ub
        mlir::Value v = firOpBuilder.create<mlir::arith::AddIOp>(loc, iv, step);
        vs.push_back(v);
        mlir::Value zero =
            firOpBuilder.createIntegerConstant(loc, step.getType(), 0);
        mlir::Value negativeStep = firOpBuilder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, step, zero);
        mlir::Value vLT = firOpBuilder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, v, ub);
        mlir::Value vGT = firOpBuilder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sgt, v, ub);
        mlir::Value icmpOp = firOpBuilder.create<mlir::arith::SelectOp>(
            loc, negativeStep, vLT, vGT);

        if (cmpOp) {
          cmpOp = firOpBuilder.create<mlir::arith::AndIOp>(loc, cmpOp, icmpOp);
        } else {
          cmpOp = icmpOp;
        }
      }

      auto ifOp = firOpBuilder.create<fir::IfOp>(loc, cmpOp, /*else*/ false);
      firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      for (auto [v, loopIV] : llvm::zip_equal(vs, loopIVs)) {
        assert(loopIV && "loopIV was not set");
        firOpBuilder.createStoreWithConvert(loc, v, loopIV);
      }
      lastPrivIP = firOpBuilder.saveInsertionPoint();
    } else if (mlir::isa<mlir::omp::SectionsOp>(op)) {
      // Already handled by genOMP()
    } else {
      TODO(converter.getCurrentLocation(),
           "lastprivate clause in constructs other than "
           "simd/worksharing-loop");
    }
  }
}

static const parser::CharBlock *
getSource(const semantics::SemanticsContext &semaCtx,
          const lower::pft::Evaluation &eval) {
  const parser::CharBlock *source = nullptr;

  auto ompConsVisit = [&](const parser::OpenMPConstruct &x) {
    std::visit(common::visitors{
                   [&](const parser::OpenMPSectionsConstruct &x) {
                     source = &std::get<0>(x.t).source;
                   },
                   [&](const parser::OpenMPLoopConstruct &x) {
                     source = &std::get<0>(x.t).source;
                   },
                   [&](const parser::OpenMPBlockConstruct &x) {
                     source = &std::get<0>(x.t).source;
                   },
                   [&](const parser::OpenMPCriticalConstruct &x) {
                     source = &std::get<0>(x.t).source;
                   },
                   [&](const parser::OpenMPAtomicConstruct &x) {
                     std::visit([&](const auto &x) { source = &x.source; },
                                x.u);
                   },
                   [&](const auto &x) { source = &x.source; },
               },
               x.u);
  };

  eval.visit(common::visitors{
      [&](const parser::OpenMPConstruct &x) { ompConsVisit(x); },
      [&](const parser::OpenMPDeclarativeConstruct &x) { source = &x.source; },
      [&](const parser::OmpEndLoopDirective &x) { source = &x.source; },
      [&](const auto &x) {},
  });

  return source;
}

void DataSharingProcessor::collectSymbolsInNestedRegions(
    lower::pft::Evaluation &eval, semantics::Symbol::Flag flag,
    llvm::SetVector<const semantics::Symbol *> &symbolsInNestedRegions) {
  for (lower::pft::Evaluation &nestedEval : eval.getNestedEvaluations()) {
    if (nestedEval.hasNestedEvaluations()) {
      if (nestedEval.isConstruct())
        // Recursively look for OpenMP constructs within `nestedEval`'s region
        collectSymbolsInNestedRegions(nestedEval, flag, symbolsInNestedRegions);
      else {
        converter.collectSymbolSet(nestedEval, symbolsInNestedRegions, flag,
                                   /*collectSymbols=*/true,
                                   /*collectHostAssociatedSymbols=*/false);
      }
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
    semantics::Symbol::Flag flag,
    llvm::SetVector<const semantics::Symbol *> &symbols) {
  // Collect all scopes associated with 'eval'.
  llvm::SetVector<const semantics::Scope *> clauseScopes;
  std::function<void(const semantics::Scope *)> collectScopes =
      [&](const semantics::Scope *scope) {
        clauseScopes.insert(scope);
        for (const semantics::Scope &child : scope->children())
          collectScopes(&child);
      };
  const parser::CharBlock *source =
      clauses.empty() ? getSource(semaCtx, eval) : &clauses.front().source;
  const semantics::Scope *curScope = nullptr;
  if (source && !source->empty()) {
    curScope = &semaCtx.FindScope(*source);
    collectScopes(curScope);
  }
  // Collect all symbols referenced in the evaluation being processed,
  // that matches 'flag'.
  llvm::SetVector<const semantics::Symbol *> allSymbols;
  converter.collectSymbolSet(eval, allSymbols, flag,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);

  llvm::SetVector<const semantics::Symbol *> symbolsInNestedRegions;
  collectSymbolsInNestedRegions(eval, flag, symbolsInNestedRegions);

  for (auto *symbol : allSymbols)
    if (visitor.isSymbolDefineBy(symbol, eval))
      symbolsInNestedRegions.remove(symbol);

  // Filter-out symbols that must not be privatized.
  bool collectImplicit = flag == semantics::Symbol::Flag::OmpImplicit;
  bool collectPreDetermined = flag == semantics::Symbol::Flag::OmpPreDetermined;

  auto isPrivatizable = [](const semantics::Symbol &sym) -> bool {
    return !semantics::IsProcedure(sym) &&
           !sym.GetUltimate().has<semantics::DerivedTypeDetails>() &&
           !sym.GetUltimate().has<semantics::NamelistDetails>() &&
           !semantics::IsImpliedDoIndex(sym.GetUltimate()) &&
           !semantics::IsStmtFunction(sym);
  };

  auto shouldCollectSymbol = [&](const semantics::Symbol *sym) {
    if (collectImplicit)
      return sym->test(semantics::Symbol::Flag::OmpImplicit);

    if (collectPreDetermined)
      return sym->test(semantics::Symbol::Flag::OmpPreDetermined);

    return !sym->test(semantics::Symbol::Flag::OmpImplicit) &&
           !sym->test(semantics::Symbol::Flag::OmpPreDetermined);
  };

  for (const auto *sym : allSymbols) {
    assert(curScope && "couldn't find current scope");
    if (isPrivatizable(*sym) && !symbolsInNestedRegions.contains(sym) &&
        !explicitlyPrivatizedSymbols.contains(sym) &&
        shouldCollectSymbol(sym) && clauseScopes.contains(&sym->owner())) {
      allPrivatizedSymbols.insert(sym);
      symbols.insert(sym);
    }
  }
}

void DataSharingProcessor::collectDefaultSymbols() {
  using DataSharingAttribute = omp::clause::Default::DataSharingAttribute;
  for (const omp::Clause &clause : clauses) {
    if (const auto *defaultClause =
            std::get_if<omp::clause::Default>(&clause.u)) {
      if (defaultClause->v == DataSharingAttribute::Private)
        collectSymbols(semantics::Symbol::Flag::OmpPrivate, defaultSymbols);
      else if (defaultClause->v == DataSharingAttribute::Firstprivate)
        collectSymbols(semantics::Symbol::Flag::OmpFirstPrivate,
                       defaultSymbols);
    }
  }
}

void DataSharingProcessor::collectImplicitSymbols() {
  // There will be no implicit symbols when a default clause is present.
  if (defaultSymbols.empty())
    collectSymbols(semantics::Symbol::Flag::OmpImplicit, implicitSymbols);
}

void DataSharingProcessor::collectPreDeterminedSymbols() {
  if (shouldCollectPreDeterminedSymbols)
    collectSymbols(semantics::Symbol::Flag::OmpPreDetermined,
                   preDeterminedSymbols);
}

void DataSharingProcessor::privatize(mlir::omp::PrivateClauseOps *clauseOps) {
  for (const semantics::Symbol *sym : allPrivatizedSymbols) {
    if (const auto *commonDet =
            sym->detailsIf<semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects())
        doPrivatize(&*mem, clauseOps);
    } else
      doPrivatize(sym, clauseOps);
  }
}

void DataSharingProcessor::copyLastPrivatize(mlir::Operation *op) {
  insertLastPrivateCompare(op);
  for (const semantics::Symbol *sym : allPrivatizedSymbols)
    if (const auto *commonDet =
            sym->detailsIf<semantics::CommonBlockDetails>()) {
      for (const auto &mem : commonDet->objects()) {
        copyLastPrivateSymbol(&*mem, &lastPrivIP);
      }
    } else {
      copyLastPrivateSymbol(sym, &lastPrivIP);
    }
}

void DataSharingProcessor::doPrivatize(const semantics::Symbol *sym,
                                       mlir::omp::PrivateClauseOps *clauseOps) {
  if (!useDelayedPrivatization) {
    cloneSymbol(sym);
    copyFirstPrivateSymbol(sym);
    return;
  }

  lower::SymbolBox hsb = converter.lookupOneLevelUpSymbol(*sym);
  assert(hsb && "Host symbol box not found");

  mlir::Type symType = hsb.getAddr().getType();
  mlir::Location symLoc = hsb.getAddr().getLoc();
  std::string privatizerName = sym->name().ToString() + ".privatizer";
  bool isFirstPrivate = sym->test(semantics::Symbol::Flag::OmpFirstPrivate);

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
    firOpBuilder.setInsertionPointToStart(moduleOp.getBody());
    auto result = firOpBuilder.create<mlir::omp::PrivateClauseOp>(
        symLoc, uniquePrivatizerName, symType,
        isFirstPrivate ? mlir::omp::DataSharingClauseType::FirstPrivate
                       : mlir::omp::DataSharingClauseType::Private);
    fir::ExtendedValue symExV = converter.getSymbolExtendedValue(*sym);
    lower::SymMapScope outerScope(symTable);

    // Populate the `alloc` region.
    {
      mlir::Region &allocRegion = result.getAllocRegion();
      mlir::Block *allocEntryBlock = firOpBuilder.createBlock(
          &allocRegion, /*insertPt=*/{}, symType, symLoc);

      firOpBuilder.setInsertionPointToEnd(allocEntryBlock);

      fir::ExtendedValue localExV =
          hlfir::translateToExtendedValue(
              symLoc, firOpBuilder, hlfir::Entity{allocRegion.getArgument(0)},
              /*contiguousHint=*/
              evaluate::IsSimplyContiguous(*sym, converter.getFoldingContext()))
              .first;

      symTable.addSymbol(*sym, localExV);
      lower::SymMapScope innerScope(symTable);
      cloneSymbol(sym);
      mlir::Value cloneAddr = symTable.shallowLookupSymbol(*sym).getAddr();
      mlir::Type cloneType = cloneAddr.getType();

      // A `convert` op is required for variables that are storage associated
      // via `equivalence`. The problem is that these variables are declared as
      // `fir.ptr`s while their privatized storage is declared as `fir.ref`,
      // therefore we convert to proper symbol type.
      mlir::Value yieldedValue =
          (symType == cloneType) ? cloneAddr
                                 : firOpBuilder.createConvert(
                                       cloneAddr.getLoc(), symType, cloneAddr);

      firOpBuilder.create<mlir::omp::YieldOp>(hsb.getAddr().getLoc(),
                                              yieldedValue);
    }

    // Populate the `copy` region if this is a `firstprivate`.
    if (isFirstPrivate) {
      mlir::Region &copyRegion = result.getCopyRegion();
      // First block argument corresponding to the original/host value while
      // second block argument corresponding to the privatized value.
      mlir::Block *copyEntryBlock = firOpBuilder.createBlock(
          &copyRegion, /*insertPt=*/{}, {symType, symType}, {symLoc, symLoc});
      firOpBuilder.setInsertionPointToEnd(copyEntryBlock);

      auto addSymbol = [&](unsigned argIdx, bool force = false) {
        symExV.match(
            [&](const fir::MutableBoxValue &box) {
              symTable.addSymbol(
                  *sym, fir::substBase(box, copyRegion.getArgument(argIdx)),
                  force);
            },
            [&](const auto &box) {
              symTable.addSymbol(*sym, copyRegion.getArgument(argIdx), force);
            });
      };

      addSymbol(0, true);
      lower::SymMapScope innerScope(symTable);
      addSymbol(1);

      auto ip = firOpBuilder.saveInsertionPoint();
      copyFirstPrivateSymbol(sym, &ip);

      firOpBuilder.create<mlir::omp::YieldOp>(
          hsb.getAddr().getLoc(), symTable.shallowLookupSymbol(*sym).getAddr());
    }

    return result;
  }();

  if (clauseOps) {
    clauseOps->privateSyms.push_back(mlir::SymbolRefAttr::get(privatizerOp));
    clauseOps->privateVars.push_back(hsb.getAddr());
  }

  symToPrivatizer[sym] = privatizerOp;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
