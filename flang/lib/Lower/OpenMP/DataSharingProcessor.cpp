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
#include "flang/Semantics/tools.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"

namespace Fortran {
namespace lower {
namespace omp {

DataSharingProcessor::DataSharingProcessor(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    const List<Clause> &clauses, lower::pft::Evaluation &eval,
    bool shouldCollectPreDeterminedSymbols, bool useDelayedPrivatization,
    lower::SymMap &symTable)
    : converter(converter), semaCtx(semaCtx),
      firOpBuilder(converter.getFirOpBuilder()), clauses(clauses), eval(eval),
      shouldCollectPreDeterminedSymbols(shouldCollectPreDeterminedSymbols),
      useDelayedPrivatization(useDelayedPrivatization), symTable(symTable) {}

DataSharingProcessor::DataSharingProcessor(lower::AbstractConverter &converter,
                                           semantics::SemanticsContext &semaCtx,
                                           lower::pft::Evaluation &eval,
                                           bool useDelayedPrivatization,
                                           lower::SymMap &symTable)
    : DataSharingProcessor(converter, semaCtx, {}, eval,
                           /*shouldCollectPreDeterminedSymols=*/false,
                           useDelayedPrivatization, symTable) {}

void DataSharingProcessor::processStep1(
    mlir::omp::PrivateClauseOps *clauseOps) {
  collectSymbolsForPrivatization();

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

static const parser::CharBlock *
getSource(const semantics::SemanticsContext &semaCtx,
          const lower::pft::Evaluation &eval) {
  const parser::CharBlock *source = nullptr;

  auto ompConsVisit = [&](const parser::OpenMPConstruct &x) {
    std::visit(
        common::visitors{
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
              source = &std::get<parser::OmpDirectiveSpecification>(x.t).source;
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

static std::optional<llvm::omp::Directive>
getDirective(lower::pft::Evaluation &eval) {
  return eval.visit([=](auto &&s) -> std::optional<llvm::omp::Directive> {
    using BareS = llvm::remove_cvref_t<decltype(s)>;
    if constexpr (std::is_same_v<BareS, parser::OpenMPConstruct>) {
      return parser::omp::GetOmpDirectiveName(s).v;
    } else {
      return std::nullopt;
    }
  });
}

const semantics::Scope *DataSharingProcessor::getCurrentScope() const {
  const parser::CharBlock *source =
      clauses.empty() ? getSource(semaCtx, eval) : &clauses.front().source;
  return source && !source->empty() ? &semaCtx.FindScope(*source) : nullptr;
}

static const semantics::Symbol *
getCurScopeSymbolAncestorRec(const semantics::Scope *curScope,
                             const semantics::Symbol *sym) {
  const semantics::Symbol *parent = nullptr;
  if (const auto *details =
          sym->detailsIf<Fortran::semantics::HostAssocDetails>())
    parent = &details->symbol();
  if (!parent)
    return nullptr;

  if (parent->owner() == *curScope)
    return parent;
  return getCurScopeSymbolAncestorRec(curScope, parent);
}

// Get the ancestor of `sym` in the current scope, if any.
static const semantics::Symbol *
getCurScopeSymbolAncestor(const semantics::Scope *curScope,
                          const semantics::Symbol *sym) {
  if (curScope == nullptr || sym->owner() == *curScope)
    return nullptr;
  return getCurScopeSymbolAncestorRec(curScope, sym);
}

void DataSharingProcessor::collectSymbolsForPrivatization() {
  using namespace Fortran::semantics;

  std::optional<llvm::omp::Directive> currentDirective = getDirective(eval);
  llvm::SetVector<const Symbol *> explicitSymbols;
  bool hasDefaultClause = false;

  // Collect explicitly privatized symbols from clauses.
  // This is needed for combined/composite constructs, in order to identify
  // which directive should privatize which symbol.
  // It is also needed for symbols that are not referenced in the construct.
  for (const omp::Clause &clause : clauses) {
    if (const auto &privateClause =
            std::get_if<omp::clause::Private>(&clause.u)) {
      collectOmpObjectListSymbol(privateClause->v, explicitSymbols);
    } else if (const auto &firstPrivateClause =
                   std::get_if<omp::clause::Firstprivate>(&clause.u)) {
      collectOmpObjectListSymbol(firstPrivateClause->v, explicitSymbols);
    } else if (const auto &lastPrivateClause =
                   std::get_if<omp::clause::Lastprivate>(&clause.u)) {
      lastprivateModifierNotSupported(*lastPrivateClause,
                                      converter.getCurrentLocation());
      const ObjectList &objects = std::get<ObjectList>(lastPrivateClause->t);
      collectOmpObjectListSymbol(objects, explicitSymbols);
    } else if (std::get_if<omp::clause::Default>(&clause.u)) {
      hasDefaultClause = true;
    }
  }

  // Filter symbols, leaving only those that must be privatized by the
  // current construct.
  auto shouldCollectSymbol = [&](const Symbol *sym) -> bool {
    // Always skip shared symbols.
    if (sym->test(Symbol::Flag::OmpShared))
      return false;

    // Explicit: skip only if linear and !pre-determined.
    //
    // The handling of linear clause is deferred to the OpenMP
    // IRBuilder which is responsible for all its aspects,
    // including privatization. Privatizing linear variables at this point would
    // cause the following structure:
    //
    // omp.op linear(%linear = %step : !fir.ref<type>) {
    // Use %linear in this BB
    // }
    //
    // to be changed to the following:
    //
    // omp. op linear(%linear = %step : !fir.ref<type>)
    //         private(%linear -> %arg0 : !fir.ref<i32>) {
    // Declare and use %arg0 in this BB
    // }
    //
    // The OpenMP IRBuilder needs to map the linear MLIR value
    // (i.e. %linear) to its `uses` in the BB to correctly
    // implement the functionalities of linear clause. However,
    // privatizing here disallows the IRBuilder to
    // draw a relation between %linear and %arg0. Hence skip, except for
    // `OmpPreDetermined` symbols, that cannot be exceptions since
    // their privatized symbols are heavily used in FIR.
    if (explicitSymbols.contains(sym)) {
      if (sym->test(Symbol::Flag::OmpLinear) &&
          !sym->test(Symbol::Flag::OmpPreDetermined))
        return false;
      return true;
    }
    // Skip explicit symbols from other directives.
    if (sym->test(Symbol::Flag::OmpExplicit))
      return false;

    // Pre-determined: collect only if flag is set.
    if (sym->test(Symbol::Flag::OmpPreDetermined))
      return shouldCollectPreDeterminedSymbols;

    // Implicit: collect if:
    // - has default clause
    // - not composite/combined
    // - it is a taskgen directive
    //   (XXX this will cause problems with "parallel ... taskloop" directives)
    if (sym->test(Symbol::Flag::OmpImplicit)) {
      if (hasDefaultClause)
        return true;
      if (currentDirective && llvm::omp::isLeafConstruct(*currentDirective))
        return true;
      if (currentDirective &&
          llvm::omp::taskGeneratingSet.test(*currentDirective))
        return true;
      return false;
    }

    // Collect everything else (unreachable?).
    return true;
  };

  // Collect symbols where `flag` is set and that match `type`.
  auto collect = [&](Symbol::Flag flag,
                     std::optional<Symbol::Flag> type = std::nullopt) {
    llvm::SetVector<const Symbol *> symbols;
    collectSymbols(flag, symbols);

    for (auto *sym : symbols) {
      if (type && !sym->test(*type))
        continue;
      if (shouldCollectSymbol(sym))
        allPrivatizedSymbols.insert(sym);
    }
  };

  // Insert explicit symbols.
  for (auto *sym : explicitSymbols)
    if (shouldCollectSymbol(sym))
      allPrivatizedSymbols.insert(sym);

  // For now, collect symbols in the same order as before, to avoid having to
  // change too many tests:
  // - implicit: private, firstprivate
  // - pre-determined
  // In the future, it should be possible to collect implicit symbols in a
  // single pass.
  collect(Symbol::Flag::OmpPrivate, Symbol::Flag::OmpImplicit);
  collect(Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpImplicit);
  collect(Symbol::Flag::OmpPreDetermined);

  // Handle implicit symbols that are shared in nested regions, but private in
  // the enclosing (current) context.
  // XXX Pre-determined symbols should probably be considered too.
  llvm::SetVector<const Symbol *> symbols;
  collectSymbolsInNestedRegions(eval, Symbol::Flag::OmpShared, symbols);

  for (auto *sym : symbols) {
    const Symbol *ancestor = getCurScopeSymbolAncestor(getCurrentScope(), sym);
    if (ancestor && ancestor->test(Symbol::Flag::OmpImplicit) &&
        !ancestor->test(Symbol::Flag::OmpShared))
      // This may result in additional privatization for non-immediate children,
      // but it is no incorrect.
      allPrivatizedSymbols.insert(ancestor);
  }
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

static void collectPrivatizingConstructs(
    llvm::SmallSet<llvm::omp::Directive, 16> &constructs, unsigned version) {
  using Clause = llvm::omp::Clause;
  using Directive = llvm::omp::Directive;

  static const Clause privatizingClauses[] = {
      Clause::OMPC_private,
      Clause::OMPC_lastprivate,
      Clause::OMPC_firstprivate,
      Clause::OMPC_in_reduction,
      Clause::OMPC_reduction,
      Clause::OMPC_linear,
      // TODO: Clause::OMPC_induction,
      Clause::OMPC_task_reduction,
      Clause::OMPC_detach,
      Clause::OMPC_use_device_ptr,
      Clause::OMPC_is_device_ptr,
  };

  for (auto dir : llvm::enum_seq_inclusive<Directive>(Directive::First_,
                                                      Directive::Last_)) {
    bool allowsPrivatizing = llvm::any_of(privatizingClauses, [&](Clause cls) {
      return llvm::omp::isAllowedClauseForDirective(dir, cls, version);
    });
    if (allowsPrivatizing)
      constructs.insert(dir);
  }
}

bool DataSharingProcessor::isOpenMPPrivatizingConstruct(
    const parser::OpenMPConstruct &omp, unsigned version) {
  static llvm::SmallSet<llvm::omp::Directive, 16> privatizing;
  [[maybe_unused]] static bool init =
      (collectPrivatizingConstructs(privatizing, version), true);

  // As of OpenMP 6.0, privatizing constructs (with the test being if they
  // allow a privatizing clause) are: dispatch, distribute, do, for, loop,
  // parallel, scope, sections, simd, single, target, target_data, task,
  // taskgroup, taskloop, and teams.
  return llvm::is_contained(privatizing,
                            parser::omp::GetOmpDirectiveName(omp).v);
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

// Collect symbols to be privatized.
// Only symbols owned by the OpenMP construct being processed are collected.
void DataSharingProcessor::collectSymbols(
    semantics::Symbol::Flag flag,
    llvm::SetVector<const semantics::Symbol *> &symbols) {
  const semantics::Scope *curScope = getCurrentScope();
  // Collect all symbols referenced in the evaluation being processed,
  // that matches 'flag'.
  llvm::SetVector<const semantics::Symbol *> allSymbols;
  converter.collectSymbolSet(eval, allSymbols, flag,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/false);

  // Filter-out symbols that must not be privatized.
  auto isPrivatizable = [](const semantics::Symbol &sym) -> bool {
    return (semantics::IsProcedurePointer(sym) ||
            !semantics::IsProcedure(sym)) &&
           !sym.GetUltimate().has<semantics::DerivedTypeDetails>() &&
           !sym.GetUltimate().has<semantics::NamelistDetails>() &&
           !semantics::IsImpliedDoIndex(sym.GetUltimate()) &&
           !semantics::IsStmtFunction(sym);
  };

  // NOTE Checking only if the symbol owner matches the current scope is
  //      not always correct. For instance, symbols referenced only inside
  //      a block statement will fail this test.
  for (const auto *sym : allSymbols) {
    assert(curScope && "couldn't find current scope");
    if (isPrivatizable(*sym) && sym->owner() == *curScope)
      symbols.insert(sym);
  }
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
