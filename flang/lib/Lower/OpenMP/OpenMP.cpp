//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
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

#include "flang/Lower/OpenMP.h"

#include "ClauseProcessor.h"
#include "DataSharingProcessor.h"
#include "DirectivesCommon.h"
#include "ReductionProcessor.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace Fortran::lower::omp;

//===----------------------------------------------------------------------===//
// Code generation helper functions
//===----------------------------------------------------------------------===//

static Fortran::lower::pft::Evaluation *
getCollapsedLoopEval(Fortran::lower::pft::Evaluation &eval, int collapseValue) {
  // Return the Evaluation of the innermost collapsed loop, or the current one
  // if there was no COLLAPSE.
  if (collapseValue == 0)
    return &eval;

  Fortran::lower::pft::Evaluation *curEval = &eval.getFirstNestedEvaluation();
  for (int i = 1; i < collapseValue; i++) {
    // The nested evaluations should be DoConstructs (i.e. they should form
    // a loop nest). Each DoConstruct is a tuple <NonLabelDoStmt, Block,
    // EndDoStmt>.
    assert(curEval->isA<Fortran::parser::DoConstruct>());
    curEval = &*std::next(curEval->getNestedEvaluations().begin());
  }
  return curEval;
}

static void genNestedEvaluations(Fortran::lower::AbstractConverter &converter,
                                 Fortran::lower::pft::Evaluation &eval,
                                 int collapseValue = 0) {
  Fortran::lower::pft::Evaluation *curEval =
      getCollapsedLoopEval(eval, collapseValue);

  for (Fortran::lower::pft::Evaluation &e : curEval->getNestedEvaluations())
    converter.genEval(e);
}

static fir::GlobalOp globalInitialization(
    Fortran::lower::AbstractConverter &converter,
    fir::FirOpBuilder &firOpBuilder, const Fortran::semantics::Symbol &sym,
    const Fortran::lower::pft::Variable &var, mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym)) {
    mlir::Type baseAddrType = ty.dyn_cast<fir::BoxType>().getEleTy();
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value nullAddr =
              b.createNullConstant(currentLocation, baseAddrType);
          mlir::Value box =
              b.create<fir::EmboxOp>(currentLocation, ty, nullAddr);
          b.create<fir::HasValueOp>(currentLocation, box);
        });
  } else {
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value undef = b.create<fir::UndefOp>(currentLocation, ty);
          b.create<fir::HasValueOp>(currentLocation, undef);
        });
  }

  return global;
}

static mlir::Operation *getCompareFromReductionOp(mlir::Operation *reductionOp,
                                                  mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (mlir::Operation *compareOp = reductionOperand.getDefiningOp()) {
      if (compareOp->getOperand(0) == loadVal ||
          compareOp->getOperand(1) == loadVal)
        assert((mlir::isa<mlir::arith::CmpIOp>(compareOp) ||
                mlir::isa<mlir::arith::CmpFOp>(compareOp)) &&
               "Expected comparison not found in reduction intrinsic");
      return compareOp;
    }
  }
  return nullptr;
}

// Get the extended value for \p val by extracting additional variable
// information from \p base.
static fir::ExtendedValue getExtendedValue(fir::ExtendedValue base,
                                           mlir::Value val) {
  return base.match(
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return fir::MutableBoxValue(val, box.nonDeferredLenParams(), {});
      },
      [&](const auto &) -> fir::ExtendedValue {
        return fir::substBase(base, val);
      });
}

#ifndef NDEBUG
static bool isThreadPrivate(Fortran::lower::SymbolRef sym) {
  if (const auto *details =
          sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
    for (const auto &obj : details->objects())
      if (!obj->test(Fortran::semantics::Symbol::Flag::OmpThreadprivate))
        return false;
    return true;
  }
  return sym->test(Fortran::semantics::Symbol::Flag::OmpThreadprivate);
}
#endif

static void threadPrivatizeVars(Fortran::lower::AbstractConverter &converter,
                                Fortran::lower::pft::Evaluation &eval) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // If the symbol corresponds to the original ThreadprivateOp, use the symbol
  // value from that operation to create one ThreadprivateOp copy operation
  // inside the parallel region.
  // In some cases, however, the symbol will correspond to the original,
  // non-threadprivate variable. This can happen, for instance, with a common
  // block, declared in a separate module, used by a parent procedure and
  // privatized in its child procedure.
  auto genThreadprivateOp = [&](Fortran::lower::SymbolRef sym) -> mlir::Value {
    assert(isThreadPrivate(sym));
    mlir::Value symValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symValue.getDefiningOp();
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
      op = declOp.getMemref().getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      symValue = mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op).getSymAddr();
    return firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  };

  llvm::SetVector<const Fortran::semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(eval, threadprivateSyms,
                             Fortran::semantics::Symbol::Flag::OmpThreadprivate,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  std::set<Fortran::semantics::SourceName> threadprivateSymNames;

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const Fortran::semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    const Fortran::semantics::Symbol *sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (threadprivateSymNames.find(sym->name()) != threadprivateSymNames.end())
      continue;
    threadprivateSymNames.insert(sym->name());
    if (const Fortran::semantics::Symbol *common =
            Fortran::semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue = Fortran::lower::genCommonBlockMember(
          converter, currentLocation, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }

  firOpBuilder.restoreInsertionPoint(insPt);
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(Fortran::lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const Fortran::semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  mlir::Type tempTy = converter.genType(*sym);
  mlir::Value temp = firOpBuilder.create<fir::AllocaOp>(
      loc, tempTy, /*pinned=*/true, /*lengthParams=*/mlir::ValueRange{},
      /*shapeParams*/ mlir::ValueRange{},
      llvm::ArrayRef<mlir::NamedAttribute>{
          fir::getAdaptToByRefAttr(firOpBuilder)});
  converter.bindSymbol(*sym, temp);
  firOpBuilder.restoreInsertionPoint(insPt);
  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  mlir::Operation *storeOp = firOpBuilder.create<fir::StoreOp>(
      loc, cvtVal, converter.getSymbolAddress(*sym));
  return storeOp;
}

struct OpWithBodyGenInfo {
  /// A type for a code-gen callback function. This takes as argument the op for
  /// which the code is being generated and returns the arguments of the op's
  /// region.
  using GenOMPRegionEntryCBFn =
      std::function<llvm::SmallVector<const Fortran::semantics::Symbol *>(
          mlir::Operation *)>;

  OpWithBodyGenInfo(Fortran::lower::AbstractConverter &converter,
                    Fortran::semantics::SemanticsContext &semaCtx,
                    mlir::Location loc, Fortran::lower::pft::Evaluation &eval)
      : converter(converter), semaCtx(semaCtx), loc(loc), eval(eval) {}

  OpWithBodyGenInfo &setGenNested(bool value) {
    genNested = value;
    return *this;
  }

  OpWithBodyGenInfo &setOuterCombined(bool value) {
    outerCombined = value;
    return *this;
  }

  OpWithBodyGenInfo &setClauses(const Fortran::parser::OmpClauseList *value) {
    clauses = value;
    return *this;
  }

  OpWithBodyGenInfo &setDataSharingProcessor(DataSharingProcessor *value) {
    dsp = value;
    return *this;
  }

  OpWithBodyGenInfo &setReductions(
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *value1,
      llvm::SmallVectorImpl<mlir::Type> *value2) {
    reductionSymbols = value1;
    reductionTypes = value2;
    return *this;
  }

  OpWithBodyGenInfo &setGenRegionEntryCb(GenOMPRegionEntryCBFn value) {
    genRegionEntryCB = value;
    return *this;
  }

  /// [inout] converter to use for the clauses.
  Fortran::lower::AbstractConverter &converter;
  /// [in] Semantics context
  Fortran::semantics::SemanticsContext &semaCtx;
  /// [in] location in source code.
  mlir::Location loc;
  /// [in] current PFT node/evaluation.
  Fortran::lower::pft::Evaluation &eval;
  /// [in] whether to generate FIR for nested evaluations
  bool genNested = true;
  /// [in] is this an outer operation - prevents privatization.
  bool outerCombined = false;
  /// [in] list of clauses to process.
  const Fortran::parser::OmpClauseList *clauses = nullptr;
  /// [in] if provided, processes the construct's data-sharing attributes.
  DataSharingProcessor *dsp = nullptr;
  /// [in] if provided, list of reduction symbols
  llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *reductionSymbols =
      nullptr;
  /// [in] if provided, list of reduction types
  llvm::SmallVectorImpl<mlir::Type> *reductionTypes = nullptr;
  /// [in] if provided, emits the op's region entry. Otherwise, an emtpy block
  /// is created in the region.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
};

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]   op - the operation the body belongs to.
/// \param [in] info - options controlling code-gen for the construction.
template <typename Op>
static void createBodyOfOp(Op &op, OpWithBodyGenInfo &info) {
  fir::FirOpBuilder &firOpBuilder = info.converter.getFirOpBuilder();

  auto insertMarker = [](fir::FirOpBuilder &builder) {
    mlir::Value undef = builder.create<fir::UndefOp>(builder.getUnknownLoc(),
                                                     builder.getIndexType());
    return undef.getDefiningOp();
  };

  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  auto regionArgs =
      [&]() -> llvm::SmallVector<const Fortran::semantics::Symbol *> {
    if (info.genRegionEntryCB != nullptr) {
      return info.genRegionEntryCB(op);
    }

    firOpBuilder.createBlock(&op.getRegion());
    return {};
  }();
  // Mark the earliest insertion point.
  mlir::Operation *marker = insertMarker(firOpBuilder);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (info.eval.lowerAsUnstructured() && !info.outerCombined)
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, info.eval.getNestedEvaluations());

  // Start with privatization, so that the lowering of the nested
  // code will use the right symbols.
  constexpr bool isLoop = std::is_same_v<Op, mlir::omp::WsloopOp> ||
                          std::is_same_v<Op, mlir::omp::SimdLoopOp>;
  bool privatize = info.clauses && !info.outerCombined;

  firOpBuilder.setInsertionPoint(marker);
  std::optional<DataSharingProcessor> tempDsp;
  if (privatize) {
    if (!info.dsp) {
      tempDsp.emplace(info.converter, info.semaCtx, *info.clauses, info.eval);
      tempDsp->processStep1();
    }
  }

  if constexpr (std::is_same_v<Op, mlir::omp::ParallelOp>) {
    threadPrivatizeVars(info.converter, info.eval);
    if (info.clauses) {
      firOpBuilder.setInsertionPoint(marker);
      ClauseProcessor(info.converter, info.semaCtx, *info.clauses)
          .processCopyin();
    }
  }

  if (info.genNested) {
    // genFIR(Evaluation&) tries to patch up unterminated blocks, causing
    // a lot of complications for our approach if the terminator generation
    // is delayed past this point. Insert a temporary terminator here, then
    // delete it.
    firOpBuilder.setInsertionPointToEnd(&op.getRegion().back());
    auto *temp = Fortran::lower::genOpenMPTerminator(
        firOpBuilder, op.getOperation(), info.loc);
    firOpBuilder.setInsertionPointAfter(marker);
    genNestedEvaluations(info.converter, info.eval);
    temp->erase();
  }

  // Get or create a unique exiting block from the given region, or
  // return nullptr if there is no exiting block.
  auto getUniqueExit = [&](mlir::Region &region) -> mlir::Block * {
    // Find the blocks where the OMP terminator should go. In simple cases
    // it is the single block in the operation's region. When the region
    // is more complicated, especially with unstructured control flow, there
    // may be multiple blocks, and some of them may have non-OMP terminators
    // resulting from lowering of the code contained within the operation.
    // All the remaining blocks are potential exit points from the op's region.
    //
    // Explicit control flow cannot exit any OpenMP region (other than via
    // STOP), and that is enforced by semantic checks prior to lowering. STOP
    // statements are lowered to a function call.

    // Collect unterminated blocks.
    llvm::SmallVector<mlir::Block *> exits;
    for (mlir::Block &b : region) {
      if (b.empty() || !b.back().hasTrait<mlir::OpTrait::IsTerminator>())
        exits.push_back(&b);
    }

    if (exits.empty())
      return nullptr;
    // If there already is a unique exiting block, do not create another one.
    // Additionally, some ops (e.g. omp.sections) require only 1 block in
    // its region.
    if (exits.size() == 1)
      return exits[0];
    mlir::Block *exit = firOpBuilder.createBlock(&region);
    for (mlir::Block *b : exits) {
      firOpBuilder.setInsertionPointToEnd(b);
      firOpBuilder.create<mlir::cf::BranchOp>(info.loc, exit);
    }
    return exit;
  };

  if (auto *exitBlock = getUniqueExit(op.getRegion())) {
    firOpBuilder.setInsertionPointToEnd(exitBlock);
    auto *term = Fortran::lower::genOpenMPTerminator(
        firOpBuilder, op.getOperation(), info.loc);
    // Only insert lastprivate code when there actually is an exit block.
    // Such a block may not exist if the nested code produced an infinite
    // loop (this may not make sense in production code, but a user could
    // write that and we should handle it).
    firOpBuilder.setInsertionPoint(term);
    if (privatize) {
      if (!info.dsp) {
        assert(tempDsp.has_value());
        tempDsp->processStep2(op, isLoop);
      } else {
        if (isLoop && regionArgs.size() > 0)
          info.dsp->setLoopIV(info.converter.getSymbolAddress(*regionArgs[0]));
        info.dsp->processStep2(op, isLoop);
      }
    }
  }

  firOpBuilder.setInsertionPointAfter(marker);
  marker->erase();
}

static void genBodyOfTargetDataOp(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval, bool genNested,
    mlir::omp::TargetDataOp &dataOp, llvm::ArrayRef<mlir::Type> useDeviceTypes,
    llvm::ArrayRef<mlir::Location> useDeviceLocs,
    llvm::ArrayRef<const Fortran::semantics::Symbol *> useDeviceSymbols,
    const mlir::Location &currentLocation) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = dataOp.getRegion();

  firOpBuilder.createBlock(&region, {}, useDeviceTypes, useDeviceLocs);

  for (auto [argIndex, argSymbol] : llvm::enumerate(useDeviceSymbols)) {
    const mlir::BlockArgument &arg = region.front().getArgument(argIndex);
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*argSymbol);
    if (auto refType = arg.getType().dyn_cast<fir::ReferenceType>()) {
      if (fir::isa_builtin_cptr_type(refType.getElementType())) {
        converter.bindSymbol(*argSymbol, arg);
      } else {
        // Avoid capture of a reference to a structured binding.
        const Fortran::semantics::Symbol *sym = argSymbol;
        extVal.match(
            [&](const fir::MutableBoxValue &mbv) {
              converter.bindSymbol(
                  *sym,
                  fir::MutableBoxValue(
                      arg, fir::factory::getNonDeferredLenParams(extVal), {}));
            },
            [&](const auto &) {
              TODO(converter.getCurrentLocation(),
                   "use_device clause operand unsupported type");
            });
      }
    } else {
      TODO(converter.getCurrentLocation(),
           "use_device clause operand unsupported type");
    }
  }

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted by clean up passes since there are no uses.
  // Remembering the position for further insertion is important since
  // there are hlfir.declares inserted above while setting block arguments
  // and new code from the body should be inserted after that.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      dataOp.getOperation()->getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Set the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  if (genNested)
    genNestedEvaluations(converter, eval);
}

template <typename OpTy, typename... Args>
static OpTy genOpWithBody(OpWithBodyGenInfo &info, Args &&...args) {
  auto op = info.converter.getFirOpBuilder().create<OpTy>(
      info.loc, std::forward<Args>(args)...);
  createBodyOfOp<OpTy>(op, info);
  return op;
}

static mlir::omp::MasterOp
genMasterOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation) {
  return genOpWithBody<mlir::omp::MasterOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested),
      /*resultTypes=*/mlir::TypeRange());
}

static mlir::omp::OrderedRegionOp
genOrderedRegionOp(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval, bool genNested,
                   mlir::Location currentLocation) {
  return genOpWithBody<mlir::omp::OrderedRegionOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested),
      /*simd=*/false);
}

static mlir::omp::ParallelOp
genParallelOp(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::SymMap &symTable,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval, bool genNested,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList,
              bool outerCombined = false) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      reductionVars;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
  llvm::SmallVector<const Fortran::semantics::Symbol *> reductionSymbols;

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(clause::If::DirectiveNameModifier::Parallel, ifClauseOperand);
  cp.processNumThreads(stmtCtx, numThreadsClauseOperand);
  cp.processProcBind(procBindKindAttr);
  cp.processDefault();
  cp.processAllocate(allocatorOperands, allocateOperands);
  if (!outerCombined)
    cp.processReduction(currentLocation, reductionVars, reductionTypes,
                        reductionDeclSymbols, &reductionSymbols);

  auto reductionCallback = [&](mlir::Operation *op) {
    llvm::SmallVector<mlir::Location> locs(reductionVars.size(),
                                           currentLocation);
    auto *block = converter.getFirOpBuilder().createBlock(&op->getRegion(0), {},
                                                          reductionTypes, locs);
    for (auto [arg, prv] :
         llvm::zip_equal(reductionSymbols, block->getArguments())) {
      converter.bindSymbol(*arg, prv);
    }
    return reductionSymbols;
  };

  mlir::UnitAttr byrefAttr;
  if (ReductionProcessor::doReductionByRef(reductionVars))
    byrefAttr = converter.getFirOpBuilder().getUnitAttr();

  OpWithBodyGenInfo genInfo =
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setOuterCombined(outerCombined)
          .setClauses(&clauseList)
          .setReductions(&reductionSymbols, &reductionTypes)
          .setGenRegionEntryCb(reductionCallback);

  if (!enableDelayedPrivatization) {
    return genOpWithBody<mlir::omp::ParallelOp>(
        genInfo,
        /*resultTypes=*/mlir::TypeRange(), ifClauseOperand,
        numThreadsClauseOperand, allocateOperands, allocatorOperands,
        reductionVars,
        reductionDeclSymbols.empty()
            ? nullptr
            : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                   reductionDeclSymbols),
        procBindKindAttr, /*private_vars=*/llvm::SmallVector<mlir::Value>{},
        /*privatizers=*/nullptr, byrefAttr);
  }

  bool privatize = !outerCombined;
  DataSharingProcessor dsp(converter, semaCtx, clauseList, eval,
                           /*useDelayedPrivatization=*/true, &symTable);

  if (privatize)
    dsp.processStep1();

  const auto &delayedPrivatizationInfo = dsp.getDelayedPrivatizationInfo();

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    auto parallelOp = llvm::cast<mlir::omp::ParallelOp>(op);

    llvm::SmallVector<mlir::Location> reductionLocs(reductionVars.size(),
                                                    currentLocation);

    mlir::OperandRange privateVars = parallelOp.getPrivateVars();
    mlir::Region &region = parallelOp.getRegion();

    llvm::SmallVector<mlir::Type> privateVarTypes = reductionTypes;
    privateVarTypes.reserve(privateVarTypes.size() + privateVars.size());
    llvm::transform(privateVars, std::back_inserter(privateVarTypes),
                    [](mlir::Value v) { return v.getType(); });

    llvm::SmallVector<mlir::Location> privateVarLocs = reductionLocs;
    privateVarLocs.reserve(privateVarLocs.size() + privateVars.size());
    llvm::transform(privateVars, std::back_inserter(privateVarLocs),
                    [](mlir::Value v) { return v.getLoc(); });

    converter.getFirOpBuilder().createBlock(&region, /*insertPt=*/{},
                                            privateVarTypes, privateVarLocs);

    llvm::SmallVector<const Fortran::semantics::Symbol *> allSymbols =
        reductionSymbols;
    allSymbols.append(delayedPrivatizationInfo.symbols);
    for (auto [arg, prv] : llvm::zip_equal(allSymbols, region.getArguments())) {
      converter.bindSymbol(*arg, prv);
    }

    return allSymbols;
  };

  // TODO Merge with the reduction CB.
  genInfo.setGenRegionEntryCb(genRegionEntryCB).setDataSharingProcessor(&dsp);

  llvm::SmallVector<mlir::Attribute> privatizers(
      delayedPrivatizationInfo.privatizers.begin(),
      delayedPrivatizationInfo.privatizers.end());

  return genOpWithBody<mlir::omp::ParallelOp>(
      genInfo,
      /*resultTypes=*/mlir::TypeRange(), ifClauseOperand,
      numThreadsClauseOperand, allocateOperands, allocatorOperands,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 reductionDeclSymbols),
      procBindKindAttr, delayedPrivatizationInfo.originalAddresses,
      delayedPrivatizationInfo.privatizers.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 privatizers),
      byrefAttr);
}

static mlir::omp::SectionOp
genSectionOp(Fortran::lower::AbstractConverter &converter,
             Fortran::semantics::SemanticsContext &semaCtx,
             Fortran::lower::pft::Evaluation &eval, bool genNested,
             mlir::Location currentLocation,
             const Fortran::parser::OmpClauseList &sectionsClauseList) {
  // Currently only private/firstprivate clause is handled, and
  // all privatization is done within `omp.section` operations.
  return genOpWithBody<mlir::omp::SectionOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&sectionsClauseList));
}

static mlir::omp::SingleOp
genSingleOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &beginClauseList,
            const Fortran::parser::OmpClauseList &endClauseList) {
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  llvm::SmallVector<mlir::Value> copyPrivateVars;
  llvm::SmallVector<mlir::Attribute> copyPrivateFuncs;
  mlir::UnitAttr nowaitAttr;

  ClauseProcessor cp(converter, semaCtx, beginClauseList);
  cp.processAllocate(allocatorOperands, allocateOperands);

  ClauseProcessor ecp(converter, semaCtx, endClauseList);
  ecp.processNowait(nowaitAttr);
  ecp.processCopyPrivate(currentLocation, copyPrivateVars, copyPrivateFuncs);

  return genOpWithBody<mlir::omp::SingleOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&beginClauseList),
      allocateOperands, allocatorOperands, copyPrivateVars,
      copyPrivateFuncs.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 copyPrivateFuncs),
      nowaitAttr);
}

static mlir::omp::TaskOp
genTaskOp(Fortran::lower::AbstractConverter &converter,
          Fortran::semantics::SemanticsContext &semaCtx,
          Fortran::lower::pft::Evaluation &eval, bool genNested,
          mlir::Location currentLocation,
          const Fortran::parser::OmpClauseList &clauseList) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, finalClauseOperand, priorityClauseOperand;
  mlir::UnitAttr untiedAttr, mergeableAttr;
  llvm::SmallVector<mlir::Attribute> dependTypeOperands;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      dependOperands;

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(clause::If::DirectiveNameModifier::Task, ifClauseOperand);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processDefault();
  cp.processFinal(stmtCtx, finalClauseOperand);
  cp.processUntied(untiedAttr);
  cp.processMergeable(mergeableAttr);
  cp.processPriority(stmtCtx, priorityClauseOperand);
  cp.processDepend(dependTypeOperands, dependOperands);
  cp.processTODO<clause::InReduction, clause::Detach, clause::Affinity>(
      currentLocation, llvm::omp::Directive::OMPD_task);

  return genOpWithBody<mlir::omp::TaskOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&clauseList),
      ifClauseOperand, finalClauseOperand, untiedAttr, mergeableAttr,
      /*in_reduction_vars=*/mlir::ValueRange(),
      /*in_reductions=*/nullptr, priorityClauseOperand,
      dependTypeOperands.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 dependTypeOperands),
      dependOperands, allocateOperands, allocatorOperands);
}

static mlir::omp::TaskgroupOp
genTaskgroupOp(Fortran::lower::AbstractConverter &converter,
               Fortran::semantics::SemanticsContext &semaCtx,
               Fortran::lower::pft::Evaluation &eval, bool genNested,
               mlir::Location currentLocation,
               const Fortran::parser::OmpClauseList &clauseList) {
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processTODO<clause::TaskReduction>(currentLocation,
                                        llvm::omp::Directive::OMPD_taskgroup);
  return genOpWithBody<mlir::omp::TaskgroupOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&clauseList),
      /*task_reduction_vars=*/mlir::ValueRange(),
      /*task_reductions=*/nullptr, allocateOperands, allocatorOperands);
}

// This helper function implements the functionality of "promoting"
// non-CPTR arguments of use_device_ptr to use_device_addr
// arguments (automagic conversion of use_device_ptr ->
// use_device_addr in these cases). The way we do so currently is
// through the shuffling of operands from the devicePtrOperands to
// deviceAddrOperands where neccesary and re-organizing the types,
// locations and symbols to maintain the correct ordering of ptr/addr
// input -> BlockArg.
//
// This effectively implements some deprecated OpenMP functionality
// that some legacy applications unfortunately depend on
// (deprecated in specification version 5.2):
//
// "If a list item in a use_device_ptr clause is not of type C_PTR,
//  the behavior is as if the list item appeared in a use_device_addr
//  clause. Support for such list items in a use_device_ptr clause
//  is deprecated."
static void promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(
    llvm::SmallVectorImpl<mlir::Value> &devicePtrOperands,
    llvm::SmallVectorImpl<mlir::Value> &deviceAddrOperands,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
        &useDeviceSymbols) {
  auto moveElementToBack = [](size_t idx, auto &vector) {
    auto *iter = std::next(vector.begin(), idx);
    vector.push_back(*iter);
    vector.erase(iter);
  };

  // Iterate over our use_device_ptr list and shift all non-cptr arguments into
  // use_device_addr.
  for (auto *it = devicePtrOperands.begin(); it != devicePtrOperands.end();) {
    if (!fir::isa_builtin_cptr_type(fir::unwrapRefType(it->getType()))) {
      deviceAddrOperands.push_back(*it);
      // We have to shuffle the symbols around as well, to maintain
      // the correct Input -> BlockArg for use_device_ptr/use_device_addr.
      // NOTE: However, as map's do not seem to be included currently
      // this isn't as pertinent, but we must try to maintain for
      // future alterations. I believe the reason they are not currently
      // is that the BlockArg assign/lowering needs to be extended
      // to a greater set of types.
      auto idx = std::distance(devicePtrOperands.begin(), it);
      moveElementToBack(idx, useDeviceTypes);
      moveElementToBack(idx, useDeviceLocs);
      moveElementToBack(idx, useDeviceSymbols);
      it = devicePtrOperands.erase(it);
      continue;
    }
    ++it;
  }
}

static mlir::omp::TargetDataOp
genTargetDataOp(Fortran::lower::AbstractConverter &converter,
                Fortran::semantics::SemanticsContext &semaCtx,
                Fortran::lower::pft::Evaluation &eval, bool genNested,
                mlir::Location currentLocation,
                const Fortran::parser::OmpClauseList &clauseList) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand;
  llvm::SmallVector<mlir::Value> mapOperands, devicePtrOperands,
      deviceAddrOperands;
  llvm::SmallVector<mlir::Type> useDeviceTypes;
  llvm::SmallVector<mlir::Location> useDeviceLocs;
  llvm::SmallVector<const Fortran::semantics::Symbol *> useDeviceSymbols;

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(clause::If::DirectiveNameModifier::TargetData, ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processUseDevicePtr(devicePtrOperands, useDeviceTypes, useDeviceLocs,
                         useDeviceSymbols);
  cp.processUseDeviceAddr(deviceAddrOperands, useDeviceTypes, useDeviceLocs,
                          useDeviceSymbols);
  // This function implements the deprecated functionality of use_device_ptr
  // that allows users to provide non-CPTR arguments to it with the caveat
  // that the compiler will treat them as use_device_addr. A lot of legacy
  // code may still depend on this functionality, so we should support it
  // in some manner. We do so currently by simply shifting non-cptr operands
  // from the use_device_ptr list into the front of the use_device_addr list
  // whilst maintaining the ordering of useDeviceLocs, useDeviceSymbols and
  // useDeviceTypes to use_device_ptr/use_device_addr input for BlockArg
  // ordering.
  // TODO: Perhaps create a user provideable compiler option that will
  // re-introduce a hard-error rather than a warning in these cases.
  promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(
      devicePtrOperands, deviceAddrOperands, useDeviceTypes, useDeviceLocs,
      useDeviceSymbols);
  cp.processMap(currentLocation, llvm::omp::Directive::OMPD_target_data,
                stmtCtx, mapOperands);

  auto dataOp = converter.getFirOpBuilder().create<mlir::omp::TargetDataOp>(
      currentLocation, ifClauseOperand, deviceOperand, devicePtrOperands,
      deviceAddrOperands, mapOperands);
  genBodyOfTargetDataOp(converter, semaCtx, eval, genNested, dataOp,
                        useDeviceTypes, useDeviceLocs, useDeviceSymbols,
                        currentLocation);
  return dataOp;
}

template <typename OpTy>
static OpTy genTargetEnterExitDataUpdateOp(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    mlir::Location currentLocation,
    const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand;
  mlir::UnitAttr nowaitAttr;
  llvm::SmallVector<mlir::Value> mapOperands, dependOperands;
  llvm::SmallVector<mlir::Attribute> dependTypeOperands;

  clause::If::DirectiveNameModifier directiveName;
  // GCC 9.3.0 emits a (probably) bogus warning about an unused variable.
  [[maybe_unused]] llvm::omp::Directive directive;
  if constexpr (std::is_same_v<OpTy, mlir::omp::TargetEnterDataOp>) {
    directiveName = clause::If::DirectiveNameModifier::TargetEnterData;
    directive = llvm::omp::Directive::OMPD_target_enter_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetExitDataOp>) {
    directiveName = clause::If::DirectiveNameModifier::TargetExitData;
    directive = llvm::omp::Directive::OMPD_target_exit_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetUpdateOp>) {
    directiveName = clause::If::DirectiveNameModifier::TargetUpdate;
    directive = llvm::omp::Directive::OMPD_target_update;
  } else {
    return nullptr;
  }

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(directiveName, ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processDepend(dependTypeOperands, dependOperands);
  cp.processNowait(nowaitAttr);

  if constexpr (std::is_same_v<OpTy, mlir::omp::TargetUpdateOp>) {
    cp.processMotionClauses<clause::To>(stmtCtx, mapOperands);
    cp.processMotionClauses<clause::From>(stmtCtx, mapOperands);
  } else {
    cp.processMap(currentLocation, directive, stmtCtx, mapOperands);
  }

  return firOpBuilder.create<OpTy>(
      currentLocation, ifClauseOperand, deviceOperand,
      dependTypeOperands.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 dependTypeOperands),
      dependOperands, nowaitAttr, mapOperands);
}

// This functions creates a block for the body of the targetOp's region. It adds
// all the symbols present in mapSymbols as block arguments to this block.
static void
genBodyOfTargetOp(Fortran::lower::AbstractConverter &converter,
                  Fortran::semantics::SemanticsContext &semaCtx,
                  Fortran::lower::pft::Evaluation &eval, bool genNested,
                  mlir::omp::TargetOp &targetOp,
                  llvm::ArrayRef<mlir::Type> mapSymTypes,
                  llvm::ArrayRef<mlir::Location> mapSymLocs,
                  llvm::ArrayRef<const Fortran::semantics::Symbol *> mapSymbols,
                  const mlir::Location &currentLocation) {
  assert(mapSymTypes.size() == mapSymLocs.size());

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = targetOp.getRegion();

  auto *regionBlock =
      firOpBuilder.createBlock(&region, {}, mapSymTypes, mapSymLocs);

  // Clones the `bounds` placing them inside the target region and returns them.
  auto cloneBound = [&](mlir::Value bound) {
    if (mlir::isMemoryEffectFree(bound.getDefiningOp())) {
      mlir::Operation *clonedOp = bound.getDefiningOp()->clone();
      regionBlock->push_back(clonedOp);
      return clonedOp->getResult(0);
    }
    TODO(converter.getCurrentLocation(),
         "target map clause operand unsupported bound type");
  };

  auto cloneBounds = [cloneBound](llvm::ArrayRef<mlir::Value> bounds) {
    llvm::SmallVector<mlir::Value> clonedBounds;
    for (mlir::Value bound : bounds)
      clonedBounds.emplace_back(cloneBound(bound));
    return clonedBounds;
  };

  // Bind the symbols to their corresponding block arguments.
  for (auto [argIndex, argSymbol] : llvm::enumerate(mapSymbols)) {
    const mlir::BlockArgument &arg = region.getArgument(argIndex);
    // Avoid capture of a reference to a structured binding.
    const Fortran::semantics::Symbol *sym = argSymbol;
    // Structure component symbols don't have bindings.
    if (sym->owner().IsDerivedType())
      continue;
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*sym);
    extVal.match(
        [&](const fir::BoxValue &v) {
          converter.bindSymbol(*sym,
                               fir::BoxValue(arg, cloneBounds(v.getLBounds()),
                                             v.getExplicitParameters(),
                                             v.getExplicitExtents()));
        },
        [&](const fir::MutableBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::MutableBoxValue(arg, cloneBounds(v.getLBounds()),
                                         v.getMutableProperties()));
        },
        [&](const fir::ArrayBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::ArrayBoxValue(arg, cloneBounds(v.getExtents()),
                                       cloneBounds(v.getLBounds()),
                                       v.getSourceBox()));
        },
        [&](const fir::CharArrayBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::CharArrayBoxValue(arg, cloneBound(v.getLen()),
                                           cloneBounds(v.getExtents()),
                                           cloneBounds(v.getLBounds())));
        },
        [&](const fir::CharBoxValue &v) {
          converter.bindSymbol(*sym,
                               fir::CharBoxValue(arg, cloneBound(v.getLen())));
        },
        [&](const fir::UnboxedValue &v) { converter.bindSymbol(*sym, arg); },
        [&](const auto &) {
          TODO(converter.getCurrentLocation(),
               "target map clause operand unsupported type");
        });
  }

  // Check if cloning the bounds introduced any dependency on the outer region.
  // If so, then either clone them as well if they are MemoryEffectFree, or else
  // copy them to a new temporary and add them to the map and block_argument
  // lists and replace their uses with the new temporary.
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      if (mlir::isMemoryEffectFree(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        regionBlock->push_front(clonedOp);
        val.replaceUsesWithIf(
            clonedOp->getResult(0), [regionBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == regionBlock;
            });
      } else {
        auto savedIP = firOpBuilder.getInsertionPoint();
        firOpBuilder.setInsertionPointAfter(valOp);
        auto copyVal =
            firOpBuilder.createTemporary(val.getLoc(), val.getType());
        firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

        llvm::SmallVector<mlir::Value> bounds;
        std::stringstream name;
        firOpBuilder.setInsertionPoint(targetOp);
        mlir::Value mapOp = createMapInfoOp(
            firOpBuilder, copyVal.getLoc(), copyVal, mlir::Value{}, name.str(),
            bounds, llvm::SmallVector<mlir::Value>{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT),
            mlir::omp::VariableCaptureKind::ByCopy, copyVal.getType());
        targetOp.getMapOperandsMutable().append(mapOp);
        mlir::Value clonedValArg =
            region.addArgument(copyVal.getType(), copyVal.getLoc());
        firOpBuilder.setInsertionPointToStart(regionBlock);
        auto loadOp = firOpBuilder.create<fir::LoadOp>(clonedValArg.getLoc(),
                                                       clonedValArg);
        val.replaceUsesWithIf(
            loadOp->getResult(0), [regionBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == regionBlock;
            });
        firOpBuilder.setInsertionPoint(regionBlock, savedIP);
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted since there are not uses.
  // In the HLFIR flow there are hlfir.declares inserted above while
  // setting block arguments.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      targetOp.getOperation()->getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Create the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  if (genNested)
    genNestedEvaluations(converter, eval);
}

static mlir::omp::TargetOp
genTargetOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &clauseList,
            llvm::omp::Directive directive, bool outerCombined = false) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value ifClauseOperand, deviceOperand, threadLimitOperand;
  mlir::UnitAttr nowaitAttr;
  llvm::SmallVector<mlir::Attribute> dependTypeOperands;
  llvm::SmallVector<mlir::Value> mapOperands, dependOperands;
  llvm::SmallVector<mlir::Type> mapSymTypes;
  llvm::SmallVector<mlir::Location> mapSymLocs;
  llvm::SmallVector<const Fortran::semantics::Symbol *> mapSymbols;

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(clause::If::DirectiveNameModifier::Target, ifClauseOperand);
  cp.processDevice(stmtCtx, deviceOperand);
  cp.processThreadLimit(stmtCtx, threadLimitOperand);
  cp.processDepend(dependTypeOperands, dependOperands);
  cp.processNowait(nowaitAttr);
  cp.processMap(currentLocation, directive, stmtCtx, mapOperands, &mapSymTypes,
                &mapSymLocs, &mapSymbols);

  cp.processTODO<clause::Private, clause::Firstprivate, clause::IsDevicePtr,
                 clause::HasDeviceAddr, clause::Reduction, clause::InReduction,
                 clause::Allocate, clause::UsesAllocators, clause::Defaultmap>(
      currentLocation, llvm::omp::Directive::OMPD_target);

  // 5.8.1 Implicit Data-Mapping Attribute Rules
  // The following code follows the implicit data-mapping rules to map all the
  // symbols used inside the region that have not been explicitly mapped using
  // the map clause.
  auto captureImplicitMap = [&](const Fortran::semantics::Symbol &sym) {
    if (llvm::find(mapSymbols, &sym) == mapSymbols.end()) {
      mlir::Value baseOp = converter.getSymbolAddress(sym);
      if (!baseOp)
        if (const auto *details = sym.template detailsIf<
                                  Fortran::semantics::HostAssocDetails>()) {
          baseOp = converter.getSymbolAddress(details->symbol());
          converter.copySymbolBinding(details->symbol(), sym);
        }

      if (baseOp) {
        llvm::SmallVector<mlir::Value> bounds;
        std::stringstream name;
        fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
        name << sym.name().ToString();

        Fortran::lower::AddrAndBoundsInfo info =
            getDataOperandBaseAddr(converter, converter.getFirOpBuilder(), sym,
                                   converter.getCurrentLocation());
        if (fir::unwrapRefType(info.addr.getType()).isa<fir::BaseBoxType>())
          bounds =
              Fortran::lower::genBoundsOpsFromBox<mlir::omp::MapBoundsOp,
                                                  mlir::omp::MapBoundsType>(
                  converter.getFirOpBuilder(), converter.getCurrentLocation(),
                  converter, dataExv, info);
        if (fir::unwrapRefType(info.addr.getType()).isa<fir::SequenceType>()) {
          bool dataExvIsAssumedSize =
              Fortran::semantics::IsAssumedSizeArray(sym.GetUltimate());
          bounds = Fortran::lower::genBaseBoundsOps<mlir::omp::MapBoundsOp,
                                                    mlir::omp::MapBoundsType>(
              converter.getFirOpBuilder(), converter.getCurrentLocation(),
              converter, dataExv, dataExvIsAssumedSize);
        }

        llvm::omp::OpenMPOffloadMappingFlags mapFlag =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
        mlir::omp::VariableCaptureKind captureKind =
            mlir::omp::VariableCaptureKind::ByRef;

        mlir::Type eleType = baseOp.getType();
        if (auto refType = baseOp.getType().dyn_cast<fir::ReferenceType>())
          eleType = refType.getElementType();

        // If a variable is specified in declare target link and if device
        // type is not specified as `nohost`, it needs to be mapped tofrom
        mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
        mlir::Operation *op = mod.lookupSymbol(converter.mangleName(sym));
        auto declareTargetOp =
            llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(op);
        if (declareTargetOp && declareTargetOp.isDeclareTarget()) {
          if (declareTargetOp.getDeclareTargetCaptureClause() ==
                  mlir::omp::DeclareTargetCaptureClause::link &&
              declareTargetOp.getDeclareTargetDeviceType() !=
                  mlir::omp::DeclareTargetDeviceType::nohost) {
            mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
            mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
          }
        } else if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
          captureKind = mlir::omp::VariableCaptureKind::ByCopy;
        } else if (!fir::isa_builtin_cptr_type(eleType)) {
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        }

        mlir::Value mapOp = createMapInfoOp(
            converter.getFirOpBuilder(), baseOp.getLoc(), baseOp, mlir::Value{},
            name.str(), bounds, {},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                mapFlag),
            captureKind, baseOp.getType());

        mapOperands.push_back(mapOp);
        mapSymTypes.push_back(baseOp.getType());
        mapSymLocs.push_back(baseOp.getLoc());
        mapSymbols.push_back(&sym);
      }
    }
  };
  Fortran::lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = converter.getFirOpBuilder().create<mlir::omp::TargetOp>(
      currentLocation, ifClauseOperand, deviceOperand, threadLimitOperand,
      dependTypeOperands.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 dependTypeOperands),
      dependOperands, nowaitAttr, mapOperands);

  genBodyOfTargetOp(converter, semaCtx, eval, genNested, targetOp, mapSymTypes,
                    mapSymLocs, mapSymbols, currentLocation);

  return targetOp;
}

static mlir::omp::TeamsOp
genTeamsOp(Fortran::lower::AbstractConverter &converter,
           Fortran::semantics::SemanticsContext &semaCtx,
           Fortran::lower::pft::Evaluation &eval, bool genNested,
           mlir::Location currentLocation,
           const Fortran::parser::OmpClauseList &clauseList,
           bool outerCombined = false) {
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value numTeamsClauseOperand, ifClauseOperand, threadLimitClauseOperand;
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands,
      reductionVars;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;

  ClauseProcessor cp(converter, semaCtx, clauseList);
  cp.processIf(clause::If::DirectiveNameModifier::Teams, ifClauseOperand);
  cp.processAllocate(allocatorOperands, allocateOperands);
  cp.processDefault();
  cp.processNumTeams(stmtCtx, numTeamsClauseOperand);
  cp.processThreadLimit(stmtCtx, threadLimitClauseOperand);
  cp.processTODO<clause::Reduction>(currentLocation,
                                    llvm::omp::Directive::OMPD_teams);

  return genOpWithBody<mlir::omp::TeamsOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setOuterCombined(outerCombined)
          .setClauses(&clauseList),
      /*num_teams_lower=*/nullptr, numTeamsClauseOperand, ifClauseOperand,
      threadLimitClauseOperand, allocateOperands, allocatorOperands,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(converter.getFirOpBuilder().getContext(),
                                 reductionDeclSymbols));
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static mlir::omp::DeclareTargetDeviceType getDeclareTargetInfo(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {

  // The default capture type
  mlir::omp::DeclareTargetDeviceType deviceType =
      mlir::omp::DeclareTargetDeviceType::any;
  const auto &spec = std::get<Fortran::parser::OmpDeclareTargetSpecifier>(
      declareTargetConstruct.t);

  if (const auto *objectList{
          Fortran::parser::Unwrap<Fortran::parser::OmpObjectList>(spec.u)}) {
    ObjectList objects{makeList(*objectList, semaCtx)};
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(objects, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause);
  } else if (const auto *clauseList{
                 Fortran::parser::Unwrap<Fortran::parser::OmpClauseList>(
                     spec.u)}) {
    if (clauseList->v.empty()) {
      // Case: declare target, implicit capture of function
      symbolAndClause.emplace_back(
          mlir::omp::DeclareTargetCaptureClause::to,
          eval.getOwningProcedure()->getSubprogramSymbol());
    }

    ClauseProcessor cp(converter, semaCtx, *clauseList);
    cp.processTo(symbolAndClause);
    cp.processEnter(symbolAndClause);
    cp.processLink(symbolAndClause);
    cp.processDeviceType(deviceType);
    cp.processTODO<clause::Indirect>(converter.getCurrentLocation(),
                                     llvm::omp::Directive::OMPD_declare_target);
  }

  return deviceType;
}

static void collectDeferredDeclareTargets(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<Fortran::lower::OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  mlir::omp::DeclareTargetDeviceType devType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, symbolAndClause);
  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    if (!op) {
      deferredDeclareTarget.push_back(
          {std::get<0>(symClause), devType, std::get<1>(symClause)});
    }
  }
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct
        &declareTargetConstruct) {
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, symbolAndClause);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    if (mlir::isa_and_nonnull<mlir::func::FuncOp>(op))
      return deviceType;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// genOMP() Code generation helper functions
//===----------------------------------------------------------------------===//

static void
genOmpSimpleStandalone(Fortran::lower::AbstractConverter &converter,
                       Fortran::semantics::SemanticsContext &semaCtx,
                       Fortran::lower::pft::Evaluation &eval, bool genNested,
                       const Fortran::parser::OpenMPSimpleStandaloneConstruct
                           &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(simpleStandaloneConstruct.t);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    firOpBuilder.create<mlir::omp::BarrierOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    ClauseProcessor(converter, semaCtx, opClauseList)
        .processTODO<clause::Depend, clause::Nowait>(
            currentLocation, llvm::omp::Directive::OMPD_taskwait);
    firOpBuilder.create<mlir::omp::TaskwaitOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    firOpBuilder.create<mlir::omp::TaskyieldOp>(currentLocation);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genTargetDataOp(converter, semaCtx, eval, genNested, currentLocation,
                    opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    genTargetEnterExitDataUpdateOp<mlir::omp::TargetEnterDataOp>(
        converter, semaCtx, currentLocation, opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    genTargetEnterExitDataUpdateOp<mlir::omp::TargetExitDataOp>(
        converter, semaCtx, currentLocation, opClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    genTargetEnterExitDataUpdateOp<mlir::omp::TargetUpdateOp>(
        converter, semaCtx, currentLocation, opClauseList);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    TODO(currentLocation, "OMPD_ordered");
  }
}

static void
genOmpFlush(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval,
            const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
  llvm::SmallVector<mlir::Value, 4> operandRange;
  if (const auto &ompObjectList =
          std::get<std::optional<Fortran::parser::OmpObjectList>>(
              flushConstruct.t))
    genObjectList2(*ompObjectList, converter, operandRange);
  const auto &memOrderClause =
      std::get<std::optional<std::list<Fortran::parser::OmpMemoryOrderClause>>>(
          flushConstruct.t);
  if (memOrderClause && memOrderClause->size() > 0)
    TODO(converter.getCurrentLocation(), "Handle OmpMemoryOrderClause");
  converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
      converter.getCurrentLocation(), operandRange);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOmpSimpleStandalone(converter, semaCtx, eval,
                                   /*genNested=*/true,
                                   simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            genOmpFlush(converter, semaCtx, eval, flushConstruct);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
      },
      standaloneConstruct.u);
}

static llvm::SmallVector<const Fortran::semantics::Symbol *>
genLoopVars(mlir::Operation *op, Fortran::lower::AbstractConverter &converter,
            mlir::Location &loc,
            llvm::ArrayRef<const Fortran::semantics::Symbol *> args) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &region = op->getRegion(0);

  std::size_t loopVarTypeSize = 0;
  for (const Fortran::semantics::Symbol *arg : args)
    loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  llvm::SmallVector<mlir::Type> tiv(args.size(), loopVarType);
  llvm::SmallVector<mlir::Location> locs(args.size(), loc);
  firOpBuilder.createBlock(&region, {}, tiv, locs);
  // The argument is not currently in memory, so make a temporary for the
  // argument, and store it there, then bind that location to the argument.
  mlir::Operation *storeOp = nullptr;
  for (auto [argIndex, argSymbol] : llvm::enumerate(args)) {
    mlir::Value indexVal = fir::getBase(region.front().getArgument(argIndex));
    storeOp =
        createAndSetPrivatizedLoopVar(converter, loc, indexVal, argSymbol);
  }
  firOpBuilder.setInsertionPointAfter(storeOp);

  return llvm::SmallVector<const Fortran::semantics::Symbol *>(args);
}

static llvm::SmallVector<const Fortran::semantics::Symbol *>
genLoopAndReductionVars(
    mlir::Operation *op, Fortran::lower::AbstractConverter &converter,
    mlir::Location &loc,
    llvm::ArrayRef<const Fortran::semantics::Symbol *> loopArgs,
    llvm::ArrayRef<const Fortran::semantics::Symbol *> reductionArgs,
    llvm::ArrayRef<mlir::Type> reductionTypes) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  llvm::SmallVector<mlir::Type> blockArgTypes;
  llvm::SmallVector<mlir::Location> blockArgLocs;
  blockArgTypes.reserve(loopArgs.size() + reductionArgs.size());
  blockArgLocs.reserve(blockArgTypes.size());
  mlir::Block *entryBlock;

  if (loopArgs.size()) {
    std::size_t loopVarTypeSize = 0;
    for (const Fortran::semantics::Symbol *arg : loopArgs)
      loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
    mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
    std::fill_n(std::back_inserter(blockArgTypes), loopArgs.size(),
                loopVarType);
    std::fill_n(std::back_inserter(blockArgLocs), loopArgs.size(), loc);
  }
  if (reductionArgs.size()) {
    llvm::copy(reductionTypes, std::back_inserter(blockArgTypes));
    std::fill_n(std::back_inserter(blockArgLocs), reductionArgs.size(), loc);
  }
  entryBlock = firOpBuilder.createBlock(&op->getRegion(0), {}, blockArgTypes,
                                        blockArgLocs);
  // The argument is not currently in memory, so make a temporary for the
  // argument, and store it there, then bind that location to the argument.
  if (loopArgs.size()) {
    mlir::Operation *storeOp = nullptr;
    for (auto [argIndex, argSymbol] : llvm::enumerate(loopArgs)) {
      mlir::Value indexVal =
          fir::getBase(op->getRegion(0).front().getArgument(argIndex));
      storeOp =
          createAndSetPrivatizedLoopVar(converter, loc, indexVal, argSymbol);
    }
    firOpBuilder.setInsertionPointAfter(storeOp);
  }
  // Bind the reduction arguments to their block arguments
  for (auto [arg, prv] : llvm::zip_equal(
           reductionArgs,
           llvm::drop_begin(entryBlock->getArguments(), loopArgs.size()))) {
    converter.bindSymbol(*arg, prv);
  }

  return llvm::SmallVector<const Fortran::semantics::Symbol *>(loopArgs);
}

static void
createSimdLoop(Fortran::lower::AbstractConverter &converter,
               Fortran::semantics::SemanticsContext &semaCtx,
               Fortran::lower::pft::Evaluation &eval,
               llvm::omp::Directive ompDirective,
               const Fortran::parser::OmpClauseList &loopOpClauseList,
               mlir::Location loc) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  DataSharingProcessor dsp(converter, semaCtx, loopOpClauseList, eval);
  dsp.processStep1();

  Fortran::lower::StatementContext stmtCtx;
  mlir::Value scheduleChunkClauseOperand, ifClauseOperand;
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, reductionVars;
  llvm::SmallVector<mlir::Value> alignedVars, nontemporalVars;
  llvm::SmallVector<const Fortran::semantics::Symbol *> iv;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
  mlir::omp::ClauseOrderKindAttr orderClauseOperand;
  mlir::IntegerAttr simdlenClauseOperand, safelenClauseOperand;

  ClauseProcessor cp(converter, semaCtx, loopOpClauseList);
  cp.processCollapse(loc, eval, lowerBound, upperBound, step, iv);
  cp.processScheduleChunk(stmtCtx, scheduleChunkClauseOperand);
  cp.processReduction(loc, reductionVars, reductionTypes, reductionDeclSymbols);
  cp.processIf(clause::If::DirectiveNameModifier::Simd, ifClauseOperand);
  cp.processSimdlen(simdlenClauseOperand);
  cp.processSafelen(safelenClauseOperand);
  cp.processTODO<clause::Aligned, clause::Allocate, clause::Linear,
                 clause::Nontemporal, clause::Order>(loc, ompDirective);

  mlir::TypeRange resultType;
  auto simdLoopOp = firOpBuilder.create<mlir::omp::SimdLoopOp>(
      loc, resultType, lowerBound, upperBound, step, alignedVars,
      /*alignment_values=*/nullptr, ifClauseOperand, nontemporalVars,
      orderClauseOperand, simdlenClauseOperand, safelenClauseOperand,
      /*inclusive=*/firOpBuilder.getUnitAttr());

  auto *nestedEval = getCollapsedLoopEval(
      eval, Fortran::lower::getCollapseValue(loopOpClauseList));

  auto ivCallback = [&](mlir::Operation *op) {
    return genLoopVars(op, converter, loc, iv);
  };

  createBodyOfOp<mlir::omp::SimdLoopOp>(
      simdLoopOp, OpWithBodyGenInfo(converter, semaCtx, loc, *nestedEval)
                      .setClauses(&loopOpClauseList)
                      .setDataSharingProcessor(&dsp)
                      .setGenRegionEntryCb(ivCallback));
}

static void createWsloop(Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semaCtx,
                         Fortran::lower::pft::Evaluation &eval,
                         llvm::omp::Directive ompDirective,
                         const Fortran::parser::OmpClauseList &beginClauseList,
                         const Fortran::parser::OmpClauseList *endClauseList,
                         mlir::Location loc) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  DataSharingProcessor dsp(converter, semaCtx, beginClauseList, eval);
  dsp.processStep1();

  Fortran::lower::StatementContext stmtCtx;
  mlir::Value scheduleChunkClauseOperand;
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, reductionVars;
  llvm::SmallVector<mlir::Value> linearVars, linearStepVars;
  llvm::SmallVector<const Fortran::semantics::Symbol *> iv;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
  llvm::SmallVector<const Fortran::semantics::Symbol *> reductionSymbols;
  mlir::omp::ClauseOrderKindAttr orderClauseOperand;
  mlir::omp::ClauseScheduleKindAttr scheduleValClauseOperand;
  mlir::UnitAttr nowaitClauseOperand, byrefOperand, scheduleSimdClauseOperand;
  mlir::IntegerAttr orderedClauseOperand;
  mlir::omp::ScheduleModifierAttr scheduleModClauseOperand;

  ClauseProcessor cp(converter, semaCtx, beginClauseList);
  cp.processCollapse(loc, eval, lowerBound, upperBound, step, iv);
  cp.processScheduleChunk(stmtCtx, scheduleChunkClauseOperand);
  cp.processReduction(loc, reductionVars, reductionTypes, reductionDeclSymbols,
                      &reductionSymbols);
  cp.processTODO<clause::Linear, clause::Order>(loc, ompDirective);

  if (ReductionProcessor::doReductionByRef(reductionVars))
    byrefOperand = firOpBuilder.getUnitAttr();

  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsloopOp>(
      loc, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars,
      reductionDeclSymbols.empty()
          ? nullptr
          : mlir::ArrayAttr::get(firOpBuilder.getContext(),
                                 reductionDeclSymbols),
      scheduleValClauseOperand, scheduleChunkClauseOperand,
      /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr, nowaitClauseOperand, byrefOperand,
      orderedClauseOperand, orderClauseOperand,
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  if (cp.processOrdered(orderedClauseOperand))
    wsLoopOp.setOrderedValAttr(orderedClauseOperand);

  if (cp.processSchedule(scheduleValClauseOperand, scheduleModClauseOperand,
                         scheduleSimdClauseOperand)) {
    wsLoopOp.setScheduleValAttr(scheduleValClauseOperand);
    wsLoopOp.setScheduleModifierAttr(scheduleModClauseOperand);
    wsLoopOp.setSimdModifierAttr(scheduleSimdClauseOperand);
  }
  // In FORTRAN `nowait` clause occur at the end of `omp do` directive.
  // i.e
  // !$omp do
  // <...>
  // !$omp end do nowait
  if (endClauseList) {
    if (ClauseProcessor(converter, semaCtx, *endClauseList)
            .processNowait(nowaitClauseOperand))
      wsLoopOp.setNowaitAttr(nowaitClauseOperand);
  }

  auto *nestedEval = getCollapsedLoopEval(
      eval, Fortran::lower::getCollapseValue(beginClauseList));

  auto ivCallback = [&](mlir::Operation *op) {
    return genLoopAndReductionVars(op, converter, loc, iv, reductionSymbols,
                                   reductionTypes);
  };

  createBodyOfOp<mlir::omp::WsloopOp>(
      wsLoopOp, OpWithBodyGenInfo(converter, semaCtx, loc, *nestedEval)
                    .setClauses(&beginClauseList)
                    .setDataSharingProcessor(&dsp)
                    .setReductions(&reductionSymbols, &reductionTypes)
                    .setGenRegionEntryCb(ivCallback));
}

static void createSimdWsloop(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval, llvm::omp::Directive ompDirective,
    const Fortran::parser::OmpClauseList &beginClauseList,
    const Fortran::parser::OmpClauseList *endClauseList, mlir::Location loc) {
  ClauseProcessor cp(converter, semaCtx, beginClauseList);
  cp.processTODO<clause::Aligned, clause::Allocate, clause::Linear,
                 clause::Safelen, clause::Simdlen, clause::Order>(loc,
                                                                  ompDirective);
  // TODO: Add support for vectorization - add vectorization hints inside loop
  // body.
  // OpenMP standard does not specify the length of vector instructions.
  // Currently we safely assume that for !$omp do simd pragma the SIMD length
  // is equal to 1 (i.e. we generate standard workshare loop).
  // When support for vectorization is enabled, then we need to add handling of
  // if clause. Currently if clause can be skipped because we always assume
  // SIMD length = 1.
  createWsloop(converter, semaCtx, eval, ompDirective, beginClauseList,
               endClauseList, loc);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
  const auto &beginLoopDirective =
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t);
  const auto &loopOpClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginLoopDirective.t);
  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  const auto ompDirective =
      std::get<Fortran::parser::OmpLoopDirective>(beginLoopDirective.t).v;

  const auto *endClauseList = [&]() {
    using RetTy = const Fortran::parser::OmpClauseList *;
    if (auto &endLoopDirective =
            std::get<std::optional<Fortran::parser::OmpEndLoopDirective>>(
                loopConstruct.t)) {
      return RetTy(
          &std::get<Fortran::parser::OmpClauseList>((*endLoopDirective).t));
    }
    return RetTy();
  }();

  bool validDirective = false;
  if (llvm::omp::topTaskloopSet.test(ompDirective)) {
    validDirective = true;
    TODO(currentLocation, "Taskloop construct");
  } else {
    // Create omp.{target, teams, distribute, parallel} nested operations
    if ((llvm::omp::allTargetSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genTargetOp(converter, semaCtx, eval, /*genNested=*/false,
                  currentLocation, loopOpClauseList, ompDirective,
                  /*outerCombined=*/true);
    }
    if ((llvm::omp::allTeamsSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genTeamsOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                 loopOpClauseList,
                 /*outerCombined=*/true);
    }
    if (llvm::omp::allDistributeSet.test(ompDirective)) {
      validDirective = true;
      TODO(currentLocation, "Distribute construct");
    }
    if ((llvm::omp::allParallelSet & llvm::omp::loopConstructSet)
            .test(ompDirective)) {
      validDirective = true;
      genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/false,
                    currentLocation, loopOpClauseList,
                    /*outerCombined=*/true);
    }
  }
  if ((llvm::omp::allDoSet | llvm::omp::allSimdSet).test(ompDirective))
    validDirective = true;

  if (!validDirective) {
    TODO(currentLocation, "Unhandled loop directive (" +
                              llvm::omp::getOpenMPDirectiveName(ompDirective) +
                              ")");
  }

  if (llvm::omp::allDoSimdSet.test(ompDirective)) {
    // 2.9.3.2 Workshare SIMD construct
    createSimdWsloop(converter, semaCtx, eval, ompDirective, loopOpClauseList,
                     endClauseList, currentLocation);

  } else if (llvm::omp::allSimdSet.test(ompDirective)) {
    // 2.9.3.1 SIMD construct
    createSimdLoop(converter, semaCtx, eval, ompDirective, loopOpClauseList,
                   currentLocation);
    genOpenMPReduction(converter, semaCtx, loopOpClauseList);
  } else {
    createWsloop(converter, semaCtx, eval, ompDirective, loopOpClauseList,
                 endClauseList, currentLocation);
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::OmpEndBlockDirective>(blockConstruct.t);
  const auto &directive =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);
  const auto &beginClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  const auto &endClauseList =
      std::get<Fortran::parser::OmpClauseList>(endBlockDirective.t);

  for (const Fortran::parser::OmpClause &clause : beginClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::If>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumThreads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ProcBind>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Allocate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Default>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Final>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Priority>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Depend>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Firstprivate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Map>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDevicePtr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDeviceAddr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ThreadLimit>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumTeams>(&clause.u)) {
      TODO(clauseLocation, "OpenMP Block construct clause");
    }
  }

  for (const auto &clause : endClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Copyprivate>(&clause.u))
      TODO(clauseLocation, "OpenMP Block construct clause");
  }

  bool singleDirective = true;
  mlir::Location currentLocation = converter.genLocation(directive.source);
  switch (directive.v) {
  case llvm::omp::Directive::OMPD_master:
    genMasterOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    genOrderedRegionOp(converter, semaCtx, eval, /*genNested=*/true,
                       currentLocation);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/true,
                  currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_single:
    genSingleOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList, endClauseList);
    break;
  case llvm::omp::Directive::OMPD_target:
    genTargetOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList, directive.v);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genTargetDataOp(converter, semaCtx, eval, /*genNested=*/true,
                    currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_task:
    genTaskOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
              beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    genTaskgroupOp(converter, semaCtx, eval, /*genNested=*/true,
                   currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_teams:
    genTeamsOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
               beginClauseList,
               /*outerCombined=*/false);
    break;
  case llvm::omp::Directive::OMPD_workshare:
    // FIXME: Workshare is not a commonly used OpenMP construct, an
    // implementation for this feature will come later. For the codes
    // that use this construct, add a single construct for now.
    genSingleOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList, endClauseList);
    break;
  default:
    singleDirective = false;
    break;
  }

  if (singleDirective)
    return;

  // Codegen for combined directives
  bool combinedDirective = false;
  if ((llvm::omp::allTargetSet & llvm::omp::blockConstructSet)
          .test(directive.v)) {
    genTargetOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                beginClauseList, directive.v,
                /*outerCombined=*/true);
    combinedDirective = true;
  }
  if ((llvm::omp::allTeamsSet & llvm::omp::blockConstructSet)
          .test(directive.v)) {
    genTeamsOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
               beginClauseList);
    combinedDirective = true;
  }
  if ((llvm::omp::allParallelSet & llvm::omp::blockConstructSet)
          .test(directive.v)) {
    bool outerCombined =
        directive.v != llvm::omp::Directive::OMPD_target_parallel;
    genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/false,
                  currentLocation, beginClauseList, outerCombined);
    combinedDirective = true;
  }
  if ((llvm::omp::workShareSet & llvm::omp::blockConstructSet)
          .test(directive.v)) {
    genSingleOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                beginClauseList, endClauseList);
    combinedDirective = true;
  }
  if (!combinedDirective)
    TODO(currentLocation, "Unhandled block directive (" +
                              llvm::omp::getOpenMPDirectiveName(directive.v) +
                              ")");

  genNestedEvaluations(converter, eval);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::IntegerAttr hintClauseOp;
  std::string name;
  const Fortran::parser::OmpCriticalDirective &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  if (std::get<std::optional<Fortran::parser::Name>>(cd.t).has_value()) {
    name =
        std::get<std::optional<Fortran::parser::Name>>(cd.t).value().ToString();
  }

  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  ClauseProcessor(converter, semaCtx, clauseList).processHint(hintClauseOp);

  mlir::omp::CriticalOp criticalOp = [&]() {
    if (name.empty()) {
      return firOpBuilder.create<mlir::omp::CriticalOp>(
          currentLocation, mlir::FlatSymbolRefAttr());
    }
    mlir::ModuleOp module = firOpBuilder.getModule();
    mlir::OpBuilder modBuilder(module.getBodyRegion());
    auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(name);
    if (!global)
      global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
          currentLocation,
          mlir::StringAttr::get(firOpBuilder.getContext(), name), hintClauseOp);
    return firOpBuilder.create<mlir::omp::CriticalOp>(
        currentLocation, mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(),
                                                      global.getSymName()));
  }();
  auto genInfo = OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval);
  createBodyOfOp<mlir::omp::CriticalOp>(criticalOp, genInfo);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionsConstruct &sectionsConstruct) {
  mlir::Location currentLocation = converter.getCurrentLocation();
  llvm::SmallVector<mlir::Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitClauseOperand;
  const auto &beginSectionsDirective =
      std::get<Fortran::parser::OmpBeginSectionsDirective>(sectionsConstruct.t);
  const auto &sectionsClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginSectionsDirective.t);

  // Process clauses before optional omp.parallel, so that new variables are
  // allocated outside of the parallel region
  ClauseProcessor cp(converter, semaCtx, sectionsClauseList);
  cp.processSectionsReduction(currentLocation);
  cp.processAllocate(allocatorOperands, allocateOperands);

  llvm::omp::Directive dir =
      std::get<Fortran::parser::OmpSectionsDirective>(beginSectionsDirective.t)
          .v;

  // Parallel wrapper of PARALLEL SECTIONS construct
  if (dir == llvm::omp::Directive::OMPD_parallel_sections) {
    genParallelOp(converter, symTable, semaCtx, eval,
                  /*genNested=*/false, currentLocation, sectionsClauseList,
                  /*outerCombined=*/true);
  } else {
    const auto &endSectionsDirective =
        std::get<Fortran::parser::OmpEndSectionsDirective>(sectionsConstruct.t);
    const auto &endSectionsClauseList =
        std::get<Fortran::parser::OmpClauseList>(endSectionsDirective.t);
    ClauseProcessor(converter, semaCtx, endSectionsClauseList)
        .processNowait(nowaitClauseOperand);
  }

  // SECTIONS construct
  genOpWithBody<mlir::omp::SectionsOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(false),
      /*reduction_vars=*/mlir::ValueRange(),
      /*reductions=*/nullptr, allocateOperands, allocatorOperands,
      nowaitClauseOperand);

  const auto &sectionBlocks =
      std::get<Fortran::parser::OmpSectionBlocks>(sectionsConstruct.t);
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto ip = firOpBuilder.saveInsertionPoint();
  for (const auto &[nblock, neval] :
       llvm::zip(sectionBlocks.v, eval.getNestedEvaluations())) {
    symTable.pushScope();
    genSectionOp(converter, semaCtx, neval, /*genNested=*/true, currentLocation,
                 sectionsClauseList);
    symTable.popScope();
    firOpBuilder.restoreInsertionPoint(ip);
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OmpAtomicRead &atomicRead) {
            mlir::Location loc = converter.genLocation(atomicRead.source);
            Fortran::lower::genOmpAccAtomicRead<
                Fortran::parser::OmpAtomicRead,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicRead,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomicWrite &atomicWrite) {
            mlir::Location loc = converter.genLocation(atomicWrite.source);
            Fortran::lower::genOmpAccAtomicWrite<
                Fortran::parser::OmpAtomicWrite,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicWrite,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomic &atomicConstruct) {
            mlir::Location loc = converter.genLocation(atomicConstruct.source);
            Fortran::lower::genOmpAtomic<Fortran::parser::OmpAtomic,
                                         Fortran::parser::OmpAtomicClauseList>(
                converter, atomicConstruct, loc);
          },
          [&](const Fortran::parser::OmpAtomicUpdate &atomicUpdate) {
            mlir::Location loc = converter.genLocation(atomicUpdate.source);
            Fortran::lower::genOmpAccAtomicUpdate<
                Fortran::parser::OmpAtomicUpdate,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicUpdate,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomicCapture &atomicCapture) {
            mlir::Location loc = converter.genLocation(atomicCapture.source);
            Fortran::lower::genOmpAccAtomicCapture<
                Fortran::parser::OmpAtomicCapture,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicCapture,
                                                      loc);
          },
      },
      atomicConstruct.u);
}

static void
markDeclareTarget(mlir::Operation *op,
                  Fortran::lower::AbstractConverter &converter,
                  mlir::omp::DeclareTargetCaptureClause captureClause,
                  mlir::omp::DeclareTargetDeviceType deviceType) {
  // TODO: Add support for program local variables with declare target applied
  auto declareTargetOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(op);
  if (!declareTargetOp)
    fir::emitFatalError(
        converter.getCurrentLocation(),
        "Attempt to apply declare target on unsupported operation");

  // The function or global already has a declare target applied to it, very
  // likely through implicit capture (usage in another declare target
  // function/subroutine). It should be marked as any if it has been assigned
  // both host and nohost, else we skip, as there is no change
  if (declareTargetOp.isDeclareTarget()) {
    if (declareTargetOp.getDeclareTargetDeviceType() != deviceType)
      declareTargetOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                       captureClause);
    return;
  }

  declareTargetOp.setDeclareTarget(deviceType, captureClause);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPDeclareTargetConstruct
                       &declareTargetConstruct) {
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, symbolAndClause);

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    // Some symbols are deferred until later in the module, these are handled
    // upon finalization of the module for OpenMP inside of Bridge, so we simply
    // skip for now.
    if (!op)
      continue;

    markDeclareTarget(
        op, converter,
        std::get<mlir::omp::DeclareTargetCaptureClause>(symClause), deviceType);
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPConstruct &ompConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, symTable, semaCtx, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) {
            genOMP(converter, symTable, semaCtx, eval, sectionsConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {
            // SECTION constructs are handled as a part of SECTIONS.
            llvm_unreachable("Unexpected standalone OMP SECTION");
          },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            genOMP(converter, symTable, semaCtx, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPExecutableAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
          },
          [&](const Fortran::parser::OpenMPAllocatorsConstruct
                  &allocsConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, symTable, semaCtx, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            genOMP(converter, symTable, semaCtx, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            genOMP(converter, symTable, semaCtx, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &declarativeAllocate) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPDeclareReductionConstruct
                  &declareReductionConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareReductionConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareSimdConstruct
                  &declareSimdConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct
                  &declareTargetConstruct) {
            genOMP(converter, symTable, semaCtx, eval, declareTargetConstruct);
          },
          [&](const Fortran::parser::OpenMPRequiresConstruct
                  &requiresConstruct) {
            // Requires directives are gathered and processed in semantics and
            // then combined in the lowering bridge before triggering codegen
            // just once. Hence, there is no need to lower each individual
            // occurrence here.
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            // The directive is lowered when instantiating the variable to
            // support the case of threadprivate variable declared in module.
          },
      },
      ompDeclConstruct.u);
}

//===----------------------------------------------------------------------===//
// Public functions
//===----------------------------------------------------------------------===//

mlir::Operation *Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                                     mlir::Operation *op,
                                                     mlir::Location loc) {
  if (mlir::isa<mlir::omp::WsloopOp, mlir::omp::DeclareReductionOp,
                mlir::omp::AtomicUpdateOp, mlir::omp::SimdLoopOp>(op))
    return builder.create<mlir::omp::YieldOp>(loc);
  else
    return builder.create<mlir::omp::TerminatorOp>(loc);
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symTable,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &omp) {
  symTable.pushScope();
  genOMP(converter, symTable, semaCtx, eval, omp);
  symTable.popScope();
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symTable,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &omp) {
  genOMP(converter, symTable, semaCtx, eval, omp);
  genNestedEvaluations(converter, eval);
}

void Fortran::lower::genOpenMPSymbolProperties(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  assert(var.hasSymbol() && "Expecting Symbol");
  const Fortran::semantics::Symbol &sym = var.getSymbol();

  if (sym.test(Fortran::semantics::Symbol::Flag::OmpThreadprivate))
    Fortran::lower::genThreadprivateOp(converter, var);

  if (sym.test(Fortran::semantics::Symbol::Flag::OmpDeclareTarget))
    Fortran::lower::genDeclareTargetIntGlobal(converter, var);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

void Fortran::lower::genThreadprivateOp(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const Fortran::semantics::Symbol *common =
          Fortran::semantics::FindCommonBlockContaining(sym.GetUltimate())) {
    mlir::Value commonValue = converter.getSymbolAddress(*common);
    if (mlir::isa<mlir::omp::ThreadprivateOp>(commonValue.getDefiningOp())) {
      // Generate ThreadprivateOp for a common block instead of its members and
      // only do it once for a common block.
      return;
    }
    // Generate ThreadprivateOp and rebind the common block.
    mlir::Value commonThreadprivateValue =
        firOpBuilder.create<mlir::omp::ThreadprivateOp>(
            currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue = genCommonBlockMember(converter, currentLocation,
                                                 sym, commonThreadprivateValue);
  } else if (!var.isGlobal()) {
    // Non-global variable which can be in threadprivate directive must be one
    // variable in main program, and it has implicit SAVE attribute. Take it as
    // with SAVE attribute, so to create GlobalOp for it to simplify the
    // translation to LLVM IR.
    // Avoids performing multiple globalInitializations.
    fir::GlobalOp global;
    auto module = converter.getModuleOp();
    std::string globalName = converter.mangleName(sym);
    if (module.lookupSymbol<fir::GlobalOp>(globalName))
      global = module.lookupSymbol<fir::GlobalOp>(globalName);
    else
      global = globalInitialization(converter, firOpBuilder, sym, var,
                                    currentLocation);

    mlir::Value symValue = firOpBuilder.create<fir::AddrOfOp>(
        currentLocation, global.resultType(), global.getSymbol());
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  } else {
    mlir::Value symValue = converter.getSymbolAddress(sym);

    // The symbol may be use-associated multiple times, and nothing needs to be
    // done after the original symbol is mapped to the threadprivatized value
    // for the first time. Use the threadprivatized value directly.
    mlir::Operation *op;
    if (auto declOp = symValue.getDefiningOp<hlfir::DeclareOp>())
      op = declOp.getMemref().getDefiningOp();
    else
      op = symValue.getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      return;

    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  }

  fir::ExtendedValue sexv = converter.getSymbolExtendedValue(sym);
  fir::ExtendedValue symThreadprivateExv =
      getExtendedValue(sexv, symThreadprivateValue);
  converter.bindSymbol(sym, symThreadprivateExv);
}

// This function replicates threadprivate's behaviour of generating
// an internal fir.GlobalOp for non-global variables in the main program
// that have the implicit SAVE attribute, to simplifiy LLVM-IR and MLIR
// generation.
void Fortran::lower::genDeclareTargetIntGlobal(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  if (!var.isGlobal()) {
    // A non-global variable which can be in a declare target directive must
    // be a variable in the main program, and it has the implicit SAVE
    // attribute. We create a GlobalOp for it to simplify the translation to
    // LLVM IR.
    globalInitialization(converter, converter.getFirOpBuilder(),
                         var.getSymbol(), var, converter.getCurrentLocation());
  }
}

// Generate an OpenMP reduction operation.
// TODO: Currently assumes it is either an integer addition/multiplication
// reduction, or a logical and reduction. Generalize this for various reduction
// operation types.
// TODO: Generate the reduction operation during lowering instead of creating
// and removing operations since this is not a robust approach. Also, removing
// ops in the builder (instead of a rewriter) is probably not the best approach.
void Fortran::lower::genOpenMPReduction(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  List<Clause> clauses{makeList(clauseList, semaCtx)};

  for (const Clause &clause : clauses) {
    if (const auto &reductionClause =
            std::get_if<clause::Reduction>(&clause.u)) {
      const auto &redOperator{
          std::get<clause::ReductionOperator>(reductionClause->t)};
      const auto &objects{std::get<ObjectList>(reductionClause->t)};
      if (const auto *reductionOp =
              std::get_if<clause::DefinedOperator>(&redOperator.u)) {
        const auto &intrinsicOp{
            std::get<clause::DefinedOperator::IntrinsicOperator>(
                reductionOp->u)};

        switch (intrinsicOp) {
        case clause::DefinedOperator::IntrinsicOperator::Add:
        case clause::DefinedOperator::IntrinsicOperator::Multiply:
        case clause::DefinedOperator::IntrinsicOperator::AND:
        case clause::DefinedOperator::IntrinsicOperator::EQV:
        case clause::DefinedOperator::IntrinsicOperator::OR:
        case clause::DefinedOperator::IntrinsicOperator::NEQV:
          break;
        default:
          continue;
        }
        for (const Object &object : objects) {
          if (const Fortran::semantics::Symbol *symbol = object.id()) {
            mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
            if (auto declOp = reductionVal.getDefiningOp<hlfir::DeclareOp>())
              reductionVal = declOp.getBase();
            mlir::Type reductionType =
                reductionVal.getType().cast<fir::ReferenceType>().getEleTy();
            if (!reductionType.isa<fir::LogicalType>()) {
              if (!reductionType.isIntOrIndexOrFloat())
                continue;
            }
            for (mlir::OpOperand &reductionValUse : reductionVal.getUses()) {
              if (auto loadOp =
                      mlir::dyn_cast<fir::LoadOp>(reductionValUse.getOwner())) {
                mlir::Value loadVal = loadOp.getRes();
                if (reductionType.isa<fir::LogicalType>()) {
                  mlir::Operation *reductionOp = findReductionChain(loadVal);
                  fir::ConvertOp convertOp =
                      getConvertFromReductionOp(reductionOp, loadVal);
                  updateReduction(reductionOp, firOpBuilder, loadVal,
                                  reductionVal, &convertOp);
                  removeStoreOp(reductionOp, reductionVal);
                } else if (mlir::Operation *reductionOp =
                               findReductionChain(loadVal, &reductionVal)) {
                  updateReduction(reductionOp, firOpBuilder, loadVal,
                                  reductionVal);
                }
              }
            }
          }
        }
      } else if (const auto *reductionIntrinsic =
                     std::get_if<clause::ProcedureDesignator>(&redOperator.u)) {
        if (!ReductionProcessor::supportedIntrinsicProcReduction(
                *reductionIntrinsic))
          continue;
        ReductionProcessor::ReductionIdentifier redId =
            ReductionProcessor::getReductionType(*reductionIntrinsic);
        for (const Object &object : objects) {
          if (const Fortran::semantics::Symbol *symbol = object.id()) {
            mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
            if (auto declOp = reductionVal.getDefiningOp<hlfir::DeclareOp>())
              reductionVal = declOp.getBase();
            for (const mlir::OpOperand &reductionValUse :
                 reductionVal.getUses()) {
              if (auto loadOp =
                      mlir::dyn_cast<fir::LoadOp>(reductionValUse.getOwner())) {
                mlir::Value loadVal = loadOp.getRes();
                // Max is lowered as a compare -> select.
                // Match the pattern here.
                mlir::Operation *reductionOp =
                    findReductionChain(loadVal, &reductionVal);
                if (reductionOp == nullptr)
                  continue;

                if (redId == ReductionProcessor::ReductionIdentifier::MAX ||
                    redId == ReductionProcessor::ReductionIdentifier::MIN) {
                  assert(mlir::isa<mlir::arith::SelectOp>(reductionOp) &&
                         "Selection Op not found in reduction intrinsic");
                  mlir::Operation *compareOp =
                      getCompareFromReductionOp(reductionOp, loadVal);
                  updateReduction(compareOp, firOpBuilder, loadVal,
                                  reductionVal);
                }
                if (redId == ReductionProcessor::ReductionIdentifier::IOR ||
                    redId == ReductionProcessor::ReductionIdentifier::IEOR ||
                    redId == ReductionProcessor::ReductionIdentifier::IAND) {
                  updateReduction(reductionOp, firOpBuilder, loadVal,
                                  reductionVal);
                }
              }
            }
          }
        }
      }
    }
  }
}

mlir::Operation *Fortran::lower::findReductionChain(mlir::Value loadVal,
                                                    mlir::Value *reductionVal) {
  for (mlir::OpOperand &loadOperand : loadVal.getUses()) {
    if (mlir::Operation *reductionOp = loadOperand.getOwner()) {
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(reductionOp)) {
        for (mlir::OpOperand &convertOperand : convertOp.getRes().getUses()) {
          if (mlir::Operation *reductionOp = convertOperand.getOwner())
            return reductionOp;
        }
      }
      for (mlir::OpOperand &reductionOperand : reductionOp->getUses()) {
        if (auto store =
                mlir::dyn_cast<fir::StoreOp>(reductionOperand.getOwner())) {
          if (store.getMemref() == *reductionVal) {
            store.erase();
            return reductionOp;
          }
        }
        if (auto assign =
                mlir::dyn_cast<hlfir::AssignOp>(reductionOperand.getOwner())) {
          if (assign.getLhs() == *reductionVal) {
            assign.erase();
            return reductionOp;
          }
        }
      }
    }
  }
  return nullptr;
}

// for a logical operator 'op' reduction X = X op Y
// This function returns the operation responsible for converting Y from
// fir.logical<4> to i1
fir::ConvertOp
Fortran::lower::getConvertFromReductionOp(mlir::Operation *reductionOp,
                                          mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (auto convertOp =
            mlir::dyn_cast<fir::ConvertOp>(reductionOperand.getDefiningOp())) {
      if (convertOp.getOperand() == loadVal)
        continue;
      return convertOp;
    }
  }
  return nullptr;
}

void Fortran::lower::updateReduction(mlir::Operation *op,
                                     fir::FirOpBuilder &firOpBuilder,
                                     mlir::Value loadVal,
                                     mlir::Value reductionVal,
                                     fir::ConvertOp *convertOp) {
  mlir::OpBuilder::InsertPoint insertPtDel = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPoint(op);

  mlir::Value reductionOp;
  if (convertOp)
    reductionOp = convertOp->getOperand();
  else if (op->getOperand(0) == loadVal)
    reductionOp = op->getOperand(1);
  else
    reductionOp = op->getOperand(0);

  firOpBuilder.create<mlir::omp::ReductionOp>(op->getLoc(), reductionOp,
                                              reductionVal);
  firOpBuilder.restoreInsertionPoint(insertPtDel);
}

void Fortran::lower::removeStoreOp(mlir::Operation *reductionOp,
                                   mlir::Value symVal) {
  for (mlir::Operation *reductionOpUse : reductionOp->getUsers()) {
    if (auto convertReduction =
            mlir::dyn_cast<fir::ConvertOp>(reductionOpUse)) {
      for (mlir::Operation *convertReductionUse :
           convertReduction.getRes().getUsers()) {
        if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(convertReductionUse)) {
          if (storeOp.getMemref() == symVal)
            storeOp.erase();
        }
        if (auto assignOp =
                mlir::dyn_cast<hlfir::AssignOp>(convertReductionUse)) {
          if (assignOp.getLhs() == symVal)
            assignOp.erase();
        }
      }
    }
  }
}

bool Fortran::lower::isOpenMPTargetConstruct(
    const Fortran::parser::OpenMPConstruct &omp) {
  llvm::omp::Directive dir = llvm::omp::Directive::OMPD_unknown;
  if (const auto *block =
          std::get_if<Fortran::parser::OpenMPBlockConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginBlockDirective>(block->t);
    dir = std::get<Fortran::parser::OmpBlockDirective>(begin.t).v;
  } else if (const auto *loop =
                 std::get_if<Fortran::parser::OpenMPLoopConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginLoopDirective>(loop->t);
    dir = std::get<Fortran::parser::OmpLoopDirective>(begin.t).v;
  }
  return llvm::omp::allTargetSet.test(dir);
}

void Fortran::lower::gatherOpenMPDeferredDeclareTargets(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct &ompReq) {
            collectDeferredDeclareTargets(converter, semaCtx, eval, ompReq,
                                          deferredDeclareTarget);
          },
          [&](const auto &) {},
      },
      ompDecl.u);
}

bool Fortran::lower::isOpenMPDeviceDeclareTarget(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl) {
  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct &ompReq) {
            mlir::omp::DeclareTargetDeviceType targetType =
                getDeclareTargetFunctionDevice(converter, semaCtx, eval, ompReq)
                    .value_or(mlir::omp::DeclareTargetDeviceType::host);
            return targetType != mlir::omp::DeclareTargetDeviceType::host;
          },
          [&](const auto &) { return false; },
      },
      ompDecl.u);
}

// In certain cases such as subroutine or function interfaces which declare
// but do not define or directly call the subroutine or function in the same
// module, their lowering is delayed until after the declare target construct
// itself is processed, so there symbol is not within the table.
//
// This function will also return true if we encounter any device declare
// target cases, to satisfy checking if we require the requires attributes
// on the module.
bool Fortran::lower::markOpenMPDeferredDeclareTargetFunctions(
    mlir::Operation *mod,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo> &deferredDeclareTargets,
    AbstractConverter &converter) {
  bool deviceCodeFound = false;
  auto modOp = llvm::cast<mlir::ModuleOp>(mod);
  for (auto declTar : deferredDeclareTargets) {
    mlir::Operation *op = modOp.lookupSymbol(converter.mangleName(declTar.sym));

    // Due to interfaces being optionally emitted on usage in a module,
    // not finding an operation at this point cannot be a hard error, we
    // simply ignore it for now.
    // TODO: Add semantic checks for detecting cases where an erronous
    // (undefined) symbol has been supplied to a declare target clause
    if (!op)
      continue;

    auto devType = declTar.declareTargetDeviceType;
    if (!deviceCodeFound && devType != mlir::omp::DeclareTargetDeviceType::host)
      deviceCodeFound = true;

    markDeclareTarget(op, converter, declTar.declareTargetCaptureClause,
                      devType);
  }

  return deviceCodeFound;
}

void Fortran::lower::genOpenMPRequires(
    mlir::Operation *mod, const Fortran::semantics::Symbol *symbol) {
  using MlirRequires = mlir::omp::ClauseRequires;
  using SemaRequires = Fortran::semantics::WithOmpDeclarative::RequiresFlag;

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    Fortran::semantics::WithOmpDeclarative::RequiresFlags semaFlags;
    if (symbol) {
      Fortran::common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<
                              Fortran::semantics::WithOmpDeclarative,
                              std::decay_t<decltype(details)>>) {
              if (details.has_ompRequires())
                semaFlags = *details.ompRequires();
            }
          },
          symbol->details());
    }

    MlirRequires mlirFlags = MlirRequires::none;
    if (semaFlags.test(SemaRequires::ReverseOffload))
      mlirFlags = mlirFlags | MlirRequires::reverse_offload;
    if (semaFlags.test(SemaRequires::UnifiedAddress))
      mlirFlags = mlirFlags | MlirRequires::unified_address;
    if (semaFlags.test(SemaRequires::UnifiedSharedMemory))
      mlirFlags = mlirFlags | MlirRequires::unified_shared_memory;
    if (semaFlags.test(SemaRequires::DynamicAllocators))
      mlirFlags = mlirFlags | MlirRequires::dynamic_allocators;

    offloadMod.setRequires(mlirFlags);
  }
}
