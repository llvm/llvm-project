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
#include "flang/Lower/Support/PrivateReductionUtils.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <variant>

namespace Fortran {
namespace lower {
namespace omp {
bool DataSharingProcessor::OMPConstructSymbolVisitor::isSymbolDefineBy(
    const semantics::Symbol *symbol, lower::pft::Evaluation &eval) const {
  return eval.visit(common::visitors{
      [&](const parser::OpenMPConstruct &functionParserNode) {
        return symDefMap.count(symbol) &&
               symDefMap.at(symbol) == ConstructPtr(&functionParserNode);
      },
      [](const auto &functionParserNode) { return false; }});
}

bool DataSharingProcessor::OMPConstructSymbolVisitor::
    isSymbolDefineByNestedDeclaration(const semantics::Symbol *symbol) const {
  return symDefMap.count(symbol) &&
         std::holds_alternative<const parser::DeclarationConstruct *>(
             symDefMap.at(symbol));
}

static bool isConstructWithTopLevelTarget(lower::pft::Evaluation &eval) {
  const auto *ompEval = eval.getIf<parser::OpenMPConstruct>();
  if (ompEval) {
    auto dir = parser::omp::GetOmpDirectiveName(*ompEval).v;
    if (llvm::omp::topTargetSet.test(dir))
      return true;
  }
  return false;
}

DataSharingProcessor::DataSharingProcessor(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, lower::pft::Evaluation &eval,
    bool shouldCollectPreDeterminedSymbols, bool useDelayedPrivatization,
    lower::SymMap &symTable, bool isTargetPrivatization)
    : converter(converter), semaCtx(semaCtx),
      firOpBuilder(converter.getFirOpBuilder()), clauses(clauses), eval(eval),
      shouldCollectPreDeterminedSymbols(shouldCollectPreDeterminedSymbols),
      useDelayedPrivatization(useDelayedPrivatization), symTable(symTable),
      isTargetPrivatization(isTargetPrivatization), visitor(semaCtx) {
  eval.visit([&](const auto &functionParserNode) {
    parser::Walk(functionParserNode, visitor);
  });
}

DataSharingProcessor::DataSharingProcessor(lower::AbstractConverter &converter,
                                           semantics::SemanticsContext &semaCtx,
                                           lower::pft::Evaluation &eval,
                                           bool useDelayedPrivatization,
                                           lower::SymMap &symTable,
                                           bool isTargetPrivatization)
    : DataSharingProcessor(converter, semaCtx, {}, eval,
                           /*shouldCollectPreDeterminedSymols=*/false,
                           useDelayedPrivatization, symTable,
                           isTargetPrivatization) {}

void DataSharingProcessor::processStep1(
    mlir::omp::PrivateClauseOps *clauseOps) {
  collectSymbolsForPrivatization();
  collectDefaultSymbols();
  collectImplicitSymbols();
  collectPreDeterminedSymbols();

  privatize(clauseOps);

  insertBarrier(clauseOps);
}

void DataSharingProcessor::processStep2(mlir::Operation *op, bool isLoop) {
  // 'sections' lastprivate is handled by genOMP()
  if (mlir::isa<mlir::omp::SectionOp>(op))
    return;
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
      // For delayed privatization deallocs are created by
      // populateByRefInitAndCleanupRegions
    }
}

void DataSharingProcessor::cloneSymbol(const semantics::Symbol *sym) {
  bool isFirstPrivate = sym->test(semantics::Symbol::Flag::OmpFirstPrivate);

  // If we are doing eager-privatization on a symbol created using delayed
  // privatization there could be incompatible types here e.g.
  // fir.ref<fir.box<fir.array<>>>
  bool success = [&]() -> bool {
    const auto *details =
        sym->detailsIf<Fortran::semantics::HostAssocDetails>();
    assert(details && "No host-association found");
    const Fortran::semantics::Symbol &hsym = details->symbol();
    mlir::Value addr = converter.getSymbolAddress(hsym);

    if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(addr.getType())) {
      if (auto boxTy = mlir::dyn_cast<fir::BoxType>(refTy.getElementType())) {
        if (auto arrayTy =
                mlir::dyn_cast<fir::SequenceType>(boxTy.getElementType())) {
          // FirConverter/fir::ExtendedValue considers all references to boxes
          // as mutable boxes. Outside of OpenMP it doesn't make sense to have a
          // mutable box of an array. Work around this here by loading the
          // reference so it is a normal boxed array.
          fir::FirOpBuilder &builder = converter.getFirOpBuilder();
          mlir::Location loc = converter.genLocation(hsym.name());
          fir::ExtendedValue hexv = converter.getSymbolExtendedValue(hsym);

          llvm::SmallVector<mlir::Value> extents =
              fir::factory::getExtents(loc, builder, hexv);

          // TODO: uniqName, name
          mlir::Value allocVal =
              builder.allocateLocal(loc, arrayTy, /*uniqName=*/"",
                                    /*name=*/"", extents, /*typeParams=*/{},
                                    sym->GetUltimate().attrs().test(
                                        Fortran::semantics::Attr::TARGET));
          mlir::Value shape = builder.genShape(loc, extents);
          mlir::Value box = builder.createBox(loc, boxTy, allocVal, shape,
                                              nullptr, {}, nullptr);

          // This can't be a CharArrayBoxValue because otherwise
          // boxTy.getElementType() would be a character type.
          // Assume the array element type isn't polymorphic because we are
          // privatizing.
          fir::ExtendedValue newExv = fir::ArrayBoxValue{box, extents};

          converter.bindSymbol(*sym, newExv);
          return true;
        }
      }
    }

    // Normal case:
    return converter.createHostAssociateVarClone(
        *sym, /*skipDefaultInit=*/isFirstPrivate);
  }();
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
    mightHaveReadHostSym.insert(sym);
  }
}

void DataSharingProcessor::copyFirstPrivateSymbol(
    const semantics::Symbol *sym, mlir::OpBuilder::InsertPoint *copyAssignIP) {
  if (sym->test(semantics::Symbol::Flag::OmpFirstPrivate) ||
      sym->test(semantics::Symbol::Flag::LocalityLocalInit))
    converter.copyHostAssociateVar(*sym, copyAssignIP);
}

void DataSharingProcessor::copyLastPrivateSymbol(
    const semantics::Symbol *sym, mlir::OpBuilder::InsertPoint *lastPrivIP) {
  if (sym->test(semantics::Symbol::Flag::OmpLastPrivate))
    converter.copyHostAssociateVar(*sym, lastPrivIP, /*hostIsSource=*/false);
}

void DataSharingProcessor::collectOmpObjectListSymbol(
    const omp::ObjectList &objects,
    llvm::SetVector<const semantics::Symbol *> &symbolSet) {
  for (const omp::Object &object : objects)
    symbolSet.insert(object.sym());
}

void DataSharingProcessor::collectSymbolsForPrivatization() {
  // Add checks here for exceptional cases where privatization is not
  // needed and be deferred to a later phase (like OpenMP IRBuilder).
  // Such cases are suggested to be clearly documented and explained
  // instead of being silently skipped
  auto isException = [&](const Fortran::semantics::Symbol *sym) -> bool {
    // `OmpPreDetermined` symbols cannot be exceptions since
    // their privatized symbols are heavily used in FIR.
    if (sym->test(Fortran::semantics::Symbol::Flag::OmpPreDetermined))
      return false;

    // The handling of linear clause is deferred to the OpenMP
    // IRBuilder which is responsible for all its aspects,
    // including privatization. Privatizing linear variables at this point would
    // cause the following structure:
    //
    // omp.op linear(%linear = %step : !fir.ref<type>) {
    //	Use %linear in this BB
    // }
    //
    // to be changed to the following:
    //
    // omp. op linear(%linear = %step : !fir.ref<type>)
    // 	private(%linear -> %arg0 : !fir.ref<i32>) {
    //	Declare and use %arg0 in this BB
    // }
    //
    // The OpenMP IRBuilder needs to map the linear MLIR value
    // (i.e. %linear) to its `uses` in the BB to correctly
    // implement the functionalities of linear clause. However,
    // privatizing here disallows the IRBuilder to
    // draw a relation between %linear and %arg0. Hence skip.
    if (sym->test(Fortran::semantics::Symbol::Flag::OmpLinear))
      return true;
    return false;
  };

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

  // TODO For common blocks, add the underlying objects within the block. Doing
  // so, we won't need to explicitly handle block objects (or forget to do
  // so).
  for (auto *sym : explicitlyPrivatizedSymbols)
    if (!isException(sym))
      allPrivatizedSymbols.insert(sym);
}

bool DataSharingProcessor::needBarrier() {
  // Emit implicit barrier to synchronize threads and avoid data races on
  // initialization of firstprivate variables and post-update of lastprivate
  // variables.
  // Emit implicit barrier for linear clause in the OpenMPIRBuilder.
  for (const semantics::Symbol *sym : allPrivatizedSymbols) {
    if (sym->test(semantics::Symbol::Flag::OmpLastPrivate) &&
        (sym->test(semantics::Symbol::Flag::OmpFirstPrivate) ||
         mightHaveReadHostSym.contains(sym)))
      return true;
  }
  return false;
}

void DataSharingProcessor::insertBarrier(
    mlir::omp::PrivateClauseOps *clauseOps) {
  if (!needBarrier())
    return;

  if (useDelayedPrivatization) {
    if (clauseOps)
      clauseOps->privateNeedsBarrier =
          mlir::UnitAttr::get(&converter.getMLIRContext());
  } else {
    mlir::omp::BarrierOp::create(firOpBuilder, converter.getCurrentLocation());
  }
}

void DataSharingProcessor::insertLastPrivateCompare(mlir::Operation *op) {
  mlir::omp::LoopNestOp loopOp;
  if (auto wrapper = mlir::dyn_cast<mlir::omp::LoopWrapperInterface>(op))
    loopOp = mlir::cast<mlir::omp::LoopNestOp>(wrapper.getWrappedLoop());

  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  bool hasLastPrivate = [&]() {
    for (const semantics::Symbol *sym : allPrivatizedSymbols) {
      if (const auto *commonDet =
              sym->detailsIf<semantics::CommonBlockDetails>()) {
        for (const auto &mem : commonDet->objects())
          if (mem->test(semantics::Symbol::Flag::OmpLastPrivate))
            return true;
      } else if (sym->test(semantics::Symbol::Flag::OmpLastPrivate))
        return true;
    }

    return false;
  }();

  if (!hasLastPrivate)
    return;

  if (mlir::isa<mlir::omp::WsloopOp>(op) || mlir::isa<mlir::omp::SimdOp>(op)) {
    mlir::omp::LoopRelatedClauseOps result;
    llvm::SmallVector<const semantics::Symbol *> iv;
    collectLoopRelatedInfo(converter, converter.getCurrentLocation(), eval,
                           clauses, result, iv);

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
    mlir::Location loc = loopOp.getLoc();
    mlir::Operation *lastOper = loopOp.getRegion().back().getTerminator();
    firOpBuilder.setInsertionPoint(lastOper);

    mlir::Value cmpOp;
    llvm::SmallVector<mlir::Value> vs;
    vs.reserve(loopOp.getIVs().size());
    for (auto [iv, ub, step] : llvm::zip_equal(
             loopOp.getIVs(), result.loopUpperBounds, result.loopSteps)) {
      // v = iv + step
      // cmp = step < 0 ? v < ub : v > ub
      mlir::Value v = mlir::arith::AddIOp::create(firOpBuilder, loc, iv, step);
      vs.push_back(v);
      mlir::Value zero =
          firOpBuilder.createIntegerConstant(loc, step.getType(), 0);
      mlir::Value negativeStep = mlir::arith::CmpIOp::create(
          firOpBuilder, loc, mlir::arith::CmpIPredicate::slt, step, zero);
      mlir::Value vLT = mlir::arith::CmpIOp::create(
          firOpBuilder, loc, mlir::arith::CmpIPredicate::slt, v, ub);
      mlir::Value vGT = mlir::arith::CmpIOp::create(
          firOpBuilder, loc, mlir::arith::CmpIPredicate::sgt, v, ub);
      mlir::Value icmpOp = mlir::arith::SelectOp::create(
          firOpBuilder, loc, negativeStep, vLT, vGT);

      if (cmpOp)
        cmpOp = mlir::arith::AndIOp::create(firOpBuilder, loc, cmpOp, icmpOp);
      else
        cmpOp = icmpOp;
    }

    auto ifOp = fir::IfOp::create(firOpBuilder, loc, cmpOp, /*else*/ false);
    firOpBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    for (auto [v, loopIV] : llvm::zip_equal(vs, loopIVs)) {
      hlfir::Entity loopIVEntity{loopIV};
      loopIVEntity =
          hlfir::derefPointersAndAllocatables(loc, firOpBuilder, loopIVEntity);
      hlfir::AssignOp::create(firOpBuilder, loc, v, loopIVEntity);
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

static parser::CharBlock getSource(const semantics::SemanticsContext &semaCtx,
                                   const lower::pft::Evaluation &eval) {
  return eval.visit(common::visitors{
      [&](const parser::OpenMPConstruct &x) {
        return parser::omp::GetOmpDirectiveName(x).source;
      },
      [&](const parser::OpenMPDeclarativeConstruct &x) { return x.source; },
      [&](const parser::OmpEndLoopDirective &x) { return x.source; },
      [&](const auto &x) { return parser::CharBlock{}; },
  });
}

bool DataSharingProcessor::isOpenMPPrivatizingConstruct(
    const parser::OpenMPConstruct &omp, unsigned version) {
  return llvm::omp::isPrivatizingConstruct(
      parser::omp::GetOmpDirectiveName(omp).v, version);
}

bool DataSharingProcessor::isOpenMPPrivatizingEvaluation(
    const pft::Evaluation &eval) const {
  unsigned version = semaCtx.langOptions().OpenMPVersion;
  return eval.visit([=](auto &&s) {
    using BareS = llvm::remove_cvref_t<decltype(s)>;
    if constexpr (std::is_same_v<BareS, parser::OpenMPConstruct>) {
      return isOpenMPPrivatizingConstruct(s, version);
    } else {
      return false;
    }
  });
}

void DataSharingProcessor::collectSymbolsInNestedRegions(
    lower::pft::Evaluation &eval, semantics::Symbol::Flag flag,
    llvm::SetVector<const semantics::Symbol *> &symbolsInNestedRegions) {
  if (!eval.hasNestedEvaluations())
    return;
  for (pft::Evaluation &nestedEval : eval.getNestedEvaluations()) {
    if (isOpenMPPrivatizingEvaluation(nestedEval)) {
      converter.collectSymbolSet(nestedEval, symbolsInNestedRegions, flag,
                                 /*collectSymbols=*/true,
                                 /*collectHostAssociatedSymbols=*/false);
    } else {
      // Recursively look for OpenMP constructs within `nestedEval`'s region
      collectSymbolsInNestedRegions(nestedEval, flag, symbolsInNestedRegions);
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
  parser::CharBlock source =
      clauses.empty() ? getSource(semaCtx, eval) : clauses.front().source;
  const semantics::Scope *curScope = nullptr;
  if (!source.empty()) {
    curScope = &semaCtx.FindScope(source);
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
    if (collectImplicit) {
      // If we're a combined construct with a target region, implicit
      // firstprivate captures, should only belong to the target region
      // and not be added/captured by later directives. Parallel regions
      // will likely want the same captures to be shared and for SIMD it's
      // illegal to have firstprivate clauses.
      if (isConstructWithTopLevelTarget(eval) && !isTargetPrivatization &&
          sym->test(semantics::Symbol::Flag::OmpFirstPrivate)) {
        return false;
      }

      // Collect implicit symbols only if they are not defined by a nested
      // `DeclarationConstruct`. If `sym` is not defined by the current OpenMP
      // evaluation then it is defined by a block nested within the OpenMP
      // construct. This, in turn, means that the private allocation for the
      // symbol will be emitted as part of the nested block and there is no need
      // to privatize it within the OpenMP construct.
      return !visitor.isSymbolDefineByNestedDeclaration(sym) &&
             sym->test(semantics::Symbol::Flag::OmpImplicit);
    }

    if (collectPreDetermined) {
      // Similar to implicit symbols, collect pre-determined symbols only if
      // they are not defined by a nested `DeclarationConstruct`
      return visitor.isSymbolDefineBy(sym, eval) &&
             !visitor.isSymbolDefineByNestedDeclaration(sym) &&
             sym->test(semantics::Symbol::Flag::OmpPreDetermined);
    }

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
        privatizeSymbol(&*mem, clauseOps);
    } else
      privatizeSymbol(sym, clauseOps);
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

void DataSharingProcessor::privatizeSymbol(
    const semantics::Symbol *symToPrivatize,
    mlir::omp::PrivateClauseOps *clauseOps) {
  if (!useDelayedPrivatization) {
    cloneSymbol(symToPrivatize);
    copyFirstPrivateSymbol(symToPrivatize);
    return;
  }

  Fortran::lower::privatizeSymbol<mlir::omp::PrivateClauseOp,
                                  mlir::omp::PrivateClauseOps>(
      converter, firOpBuilder, symTable, allPrivatizedSymbols,
      mightHaveReadHostSym, symToPrivatize, clauseOps);
}
} // namespace omp
} // namespace lower
} // namespace Fortran
