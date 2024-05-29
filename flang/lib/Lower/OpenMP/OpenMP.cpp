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
#include "Clauses.h"
#include "DataSharingProcessor.h"
#include "Decomposer.h"
#include "DirectivesCommon.h"
#include "ReductionProcessor.h"
#include "Utils.h"
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

static void genOMPDispatch(lower::AbstractConverter &converter,
                           lower::SymMap &symTable,
                           semantics::SemanticsContext &semaCtx,
                           lower::pft::Evaluation &eval, mlir::Location loc,
                           const ConstructQueue &queue,
                           ConstructQueue::iterator item);

static lower::pft::Evaluation *
getCollapsedLoopEval(lower::pft::Evaluation &eval, int collapseValue) {
  // Return the Evaluation of the innermost collapsed loop, or the current one
  // if there was no COLLAPSE.
  if (collapseValue == 0)
    return &eval;

  lower::pft::Evaluation *curEval = &eval.getFirstNestedEvaluation();
  for (int i = 1; i < collapseValue; i++) {
    // The nested evaluations should be DoConstructs (i.e. they should form
    // a loop nest). Each DoConstruct is a tuple <NonLabelDoStmt, Block,
    // EndDoStmt>.
    assert(curEval->isA<parser::DoConstruct>());
    curEval = &*std::next(curEval->getNestedEvaluations().begin());
  }
  return curEval;
}

static void genNestedEvaluations(lower::AbstractConverter &converter,
                                 lower::pft::Evaluation &eval,
                                 int collapseValue = 0) {
  lower::pft::Evaluation *curEval = getCollapsedLoopEval(eval, collapseValue);

  for (lower::pft::Evaluation &e : curEval->getNestedEvaluations())
    converter.genEval(e);
}

static fir::GlobalOp globalInitialization(lower::AbstractConverter &converter,
                                          fir::FirOpBuilder &firOpBuilder,
                                          const semantics::Symbol &sym,
                                          const lower::pft::Variable &var,
                                          mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (semantics::IsAllocatableOrObjectPointer(&sym)) {
    mlir::Type baseAddrType = mlir::dyn_cast<fir::BoxType>(ty).getEleTy();
    lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value nullAddr =
              b.createNullConstant(currentLocation, baseAddrType);
          mlir::Value box =
              b.create<fir::EmboxOp>(currentLocation, ty, nullAddr);
          b.create<fir::HasValueOp>(currentLocation, box);
        });
  } else {
    lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value undef = b.create<fir::UndefOp>(currentLocation, ty);
          b.create<fir::HasValueOp>(currentLocation, undef);
        });
  }

  return global;
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
static bool isThreadPrivate(lower::SymbolRef sym) {
  if (const auto *details = sym->detailsIf<semantics::CommonBlockDetails>()) {
    for (const auto &obj : details->objects())
      if (!obj->test(semantics::Symbol::Flag::OmpThreadprivate))
        return false;
    return true;
  }
  return sym->test(semantics::Symbol::Flag::OmpThreadprivate);
}
#endif

static void threadPrivatizeVars(lower::AbstractConverter &converter,
                                lower::pft::Evaluation &eval) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // If the symbol corresponds to the original ThreadprivateOp, use the symbol
  // value from that operation to create one ThreadprivateOp copy operation
  // inside the parallel region.
  // In some cases, however, the symbol will correspond to the original,
  // non-threadprivate variable. This can happen, for instance, with a common
  // block, declared in a separate module, used by a parent procedure and
  // privatized in its child procedure.
  auto genThreadprivateOp = [&](lower::SymbolRef sym) -> mlir::Value {
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

  llvm::SetVector<const semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(eval, threadprivateSyms,
                             semantics::Symbol::Flag::OmpThreadprivate,
                             /*collectSymbols=*/true,
                             /*collectHostAssociatedSymbols=*/true);
  std::set<semantics::SourceName> threadprivateSymNames;

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    const semantics::Symbol *sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (threadprivateSymNames.find(sym->name()) != threadprivateSymNames.end())
      continue;
    threadprivateSymNames.insert(sym->name());
    if (const semantics::Symbol *common =
            semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue = lower::genCommonBlockMember(
          converter, currentLocation, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  mlir::Type tempTy = converter.genType(*sym);

  assert(converter.isPresentShallowLookup(*sym) &&
         "Expected symbol to be in symbol table.");

  firOpBuilder.restoreInsertionPoint(insPt);
  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  mlir::Operation *storeOp = firOpBuilder.create<fir::StoreOp>(
      loc, cvtVal, converter.getSymbolAddress(*sym));
  return storeOp;
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
    mlir::omp::UseDeviceClauseOps &clauseOps,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSymbols) {
  auto moveElementToBack = [](size_t idx, auto &vector) {
    auto *iter = std::next(vector.begin(), idx);
    vector.push_back(*iter);
    vector.erase(iter);
  };

  // Iterate over our use_device_ptr list and shift all non-cptr arguments into
  // use_device_addr.
  for (auto *it = clauseOps.useDevicePtrVars.begin();
       it != clauseOps.useDevicePtrVars.end();) {
    if (!fir::isa_builtin_cptr_type(fir::unwrapRefType(it->getType()))) {
      clauseOps.useDeviceAddrVars.push_back(*it);
      // We have to shuffle the symbols around as well, to maintain
      // the correct Input -> BlockArg for use_device_ptr/use_device_addr.
      // NOTE: However, as map's do not seem to be included currently
      // this isn't as pertinent, but we must try to maintain for
      // future alterations. I believe the reason they are not currently
      // is that the BlockArg assign/lowering needs to be extended
      // to a greater set of types.
      auto idx = std::distance(clauseOps.useDevicePtrVars.begin(), it);
      moveElementToBack(idx, useDeviceTypes);
      moveElementToBack(idx, useDeviceLocs);
      moveElementToBack(idx, useDeviceSymbols);
      it = clauseOps.useDevicePtrVars.erase(it);
      continue;
    }
    ++it;
  }
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static void getDeclareTargetInfo(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    mlir::omp::DeclareTargetClauseOps &clauseOps,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause) {
  const auto &spec =
      std::get<parser::OmpDeclareTargetSpecifier>(declareTargetConstruct.t);
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    ObjectList objects{makeObjects(*objectList, semaCtx)};
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(objects, mlir::omp::DeclareTargetCaptureClause::to,
                         symbolAndClause);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    List<Clause> clauses = makeClauses(*clauseList, semaCtx);
    if (clauses.empty()) {
      // Case: declare target, implicit capture of function
      symbolAndClause.emplace_back(
          mlir::omp::DeclareTargetCaptureClause::to,
          eval.getOwningProcedure()->getSubprogramSymbol());
    }

    ClauseProcessor cp(converter, semaCtx, clauses);
    cp.processDeviceType(clauseOps);
    cp.processEnter(symbolAndClause);
    cp.processLink(symbolAndClause);
    cp.processTo(symbolAndClause);

    cp.processTODO<clause::Indirect>(converter.getCurrentLocation(),
                                     llvm::omp::Directive::OMPD_declare_target);
  }
}

static void collectDeferredDeclareTargets(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<lower::OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  mlir::omp::DeclareTargetClauseOps clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);
  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    if (!op) {
      deferredDeclareTarget.push_back({std::get<0>(symClause),
                                       clauseOps.deviceType,
                                       std::get<1>(symClause)});
    }
  }
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetClauseOps clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    if (mlir::isa_and_nonnull<mlir::func::FuncOp>(op))
      return clauseOps.deviceType;
  }

  return std::nullopt;
}

/// Set up the entry block of the given `omp.loop_nest` operation, adding a
/// block argument for each loop induction variable and allocating and
/// initializing a private value to hold each of them.
///
/// This function can also bind the symbols of any variables that should match
/// block arguments on parent loop wrapper operations attached to the same
/// loop. This allows the introduction of any necessary `hlfir.declare`
/// operations inside of the entry block of the `omp.loop_nest` operation and
/// not directly under any of the wrappers, which would invalidate them.
///
/// \param [in]          op - the loop nest operation.
/// \param [in]   converter - PFT to MLIR conversion interface.
/// \param [in]         loc - location.
/// \param [in]        args - symbols of induction variables.
/// \param [in] wrapperSyms - symbols of variables to be mapped to loop wrapper
///                           entry block arguments.
/// \param [in] wrapperArgs - entry block arguments of parent loop wrappers.
static void
genLoopVars(mlir::Operation *op, lower::AbstractConverter &converter,
            mlir::Location &loc, llvm::ArrayRef<const semantics::Symbol *> args,
            llvm::ArrayRef<const semantics::Symbol *> wrapperSyms = {},
            llvm::ArrayRef<mlir::BlockArgument> wrapperArgs = {}) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &region = op->getRegion(0);

  std::size_t loopVarTypeSize = 0;
  for (const semantics::Symbol *arg : args)
    loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  llvm::SmallVector<mlir::Type> tiv(args.size(), loopVarType);
  llvm::SmallVector<mlir::Location> locs(args.size(), loc);
  firOpBuilder.createBlock(&region, {}, tiv, locs);

  // Bind the entry block arguments of parent wrappers to the corresponding
  // symbols.
  for (auto [arg, prv] : llvm::zip_equal(wrapperSyms, wrapperArgs))
    converter.bindSymbol(*arg, prv);

  // The argument is not currently in memory, so make a temporary for the
  // argument, and store it there, then bind that location to the argument.
  mlir::Operation *storeOp = nullptr;
  for (auto [argIndex, argSymbol] : llvm::enumerate(args)) {
    mlir::Value indexVal = fir::getBase(region.front().getArgument(argIndex));
    storeOp =
        createAndSetPrivatizedLoopVar(converter, loc, indexVal, argSymbol);
  }
  firOpBuilder.setInsertionPointAfter(storeOp);
}

static void
genReductionVars(mlir::Operation *op, lower::AbstractConverter &converter,
                 mlir::Location &loc,
                 llvm::ArrayRef<const semantics::Symbol *> reductionArgs,
                 llvm::ArrayRef<mlir::Type> reductionTypes) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Location> blockArgLocs(reductionArgs.size(), loc);

  mlir::Block *entryBlock = firOpBuilder.createBlock(
      &op->getRegion(0), {}, reductionTypes, blockArgLocs);

  // Bind the reduction arguments to their block arguments.
  for (auto [arg, prv] :
       llvm::zip_equal(reductionArgs, entryBlock->getArguments())) {
    converter.bindSymbol(*arg, prv);
  }
}

static void
markDeclareTarget(mlir::Operation *op, lower::AbstractConverter &converter,
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

//===----------------------------------------------------------------------===//
// Op body generation helper structures and functions
//===----------------------------------------------------------------------===//

struct OpWithBodyGenInfo {
  /// A type for a code-gen callback function. This takes as argument the op for
  /// which the code is being generated and returns the arguments of the op's
  /// region.
  using GenOMPRegionEntryCBFn =
      std::function<llvm::SmallVector<const semantics::Symbol *>(
          mlir::Operation *)>;

  OpWithBodyGenInfo(lower::AbstractConverter &converter,
                    lower::SymMap &symTable,
                    semantics::SemanticsContext &semaCtx, mlir::Location loc,
                    lower::pft::Evaluation &eval, llvm::omp::Directive dir)
      : converter(converter), symTable(symTable), semaCtx(semaCtx), loc(loc),
        eval(eval), dir(dir) {}

  OpWithBodyGenInfo &setOuterCombined(bool value) {
    outerCombined = value;
    return *this;
  }

  OpWithBodyGenInfo &setClauses(const List<Clause> *value) {
    clauses = value;
    return *this;
  }

  OpWithBodyGenInfo &setDataSharingProcessor(DataSharingProcessor *value) {
    dsp = value;
    return *this;
  }

  OpWithBodyGenInfo &
  setReductions(llvm::SmallVectorImpl<const semantics::Symbol *> *value1,
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
  lower::AbstractConverter &converter;
  /// [in] Symbol table
  lower::SymMap &symTable;
  /// [in] Semantics context
  semantics::SemanticsContext &semaCtx;
  /// [in] location in source code.
  mlir::Location loc;
  /// [in] current PFT node/evaluation.
  lower::pft::Evaluation &eval;
  /// [in] leaf directive for which to generate the op body.
  llvm::omp::Directive dir;
  /// [in] is this an outer operation - prevents privatization.
  bool outerCombined = false;
  /// [in] list of clauses to process.
  const List<Clause> *clauses = nullptr;
  /// [in] if provided, processes the construct's data-sharing attributes.
  DataSharingProcessor *dsp = nullptr;
  /// [in] if provided, list of reduction symbols
  llvm::SmallVectorImpl<const semantics::Symbol *> *reductionSymbols = nullptr;
  /// [in] if provided, list of reduction types
  llvm::SmallVectorImpl<mlir::Type> *reductionTypes = nullptr;
  /// [in] if provided, emits the op's region entry. Otherwise, an emtpy block
  /// is created in the region.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
};

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]   op  - the operation the body belongs to.
/// \param [in] info  - options controlling code-gen for the construction.
/// \param [in] queue - work queue with nested constructs.
/// \param [in] item  - item in the queue to generate body for.
static void createBodyOfOp(mlir::Operation &op, const OpWithBodyGenInfo &info,
                           const ConstructQueue &queue,
                           ConstructQueue::iterator item) {
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
  auto regionArgs = [&]() -> llvm::SmallVector<const semantics::Symbol *> {
    if (info.genRegionEntryCB != nullptr) {
      return info.genRegionEntryCB(&op);
    }

    firOpBuilder.createBlock(&op.getRegion(0));
    return {};
  }();
  // Mark the earliest insertion point.
  mlir::Operation *marker = insertMarker(firOpBuilder);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (info.eval.lowerAsUnstructured() && !info.outerCombined)
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, info.eval.getNestedEvaluations());

  // Start with privatization, so that the lowering of the nested
  // code will use the right symbols.
  bool isLoop = llvm::omp::getDirectiveAssociation(info.dir) ==
                llvm::omp::Association::Loop;
  bool privatize = info.clauses && !info.outerCombined;

  firOpBuilder.setInsertionPoint(marker);
  std::optional<DataSharingProcessor> tempDsp;
  if (privatize) {
    if (!info.dsp) {
      tempDsp.emplace(info.converter, info.semaCtx, *info.clauses, info.eval,
                      Fortran::lower::omp::isLastItemInQueue(item, queue));
      tempDsp->processStep1();
    }
  }

  if (info.dir == llvm::omp::Directive::OMPD_parallel) {
    threadPrivatizeVars(info.converter, info.eval);
    if (info.clauses) {
      firOpBuilder.setInsertionPoint(marker);
      ClauseProcessor(info.converter, info.semaCtx, *info.clauses)
          .processCopyin();
    }
  }

  if (ConstructQueue::iterator next = std::next(item); next != queue.end()) {
    genOMPDispatch(info.converter, info.symTable, info.semaCtx, info.eval,
                   info.loc, queue, next);
  } else {
    // genFIR(Evaluation&) tries to patch up unterminated blocks, causing
    // a lot of complications for our approach if the terminator generation
    // is delayed past this point. Insert a temporary terminator here, then
    // delete it.
    firOpBuilder.setInsertionPointToEnd(&op.getRegion(0).back());
    auto *temp = lower::genOpenMPTerminator(firOpBuilder, &op, info.loc);
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

  if (auto *exitBlock = getUniqueExit(op.getRegion(0))) {
    firOpBuilder.setInsertionPointToEnd(exitBlock);
    auto *term = lower::genOpenMPTerminator(firOpBuilder, &op, info.loc);
    // Only insert lastprivate code when there actually is an exit block.
    // Such a block may not exist if the nested code produced an infinite
    // loop (this may not make sense in production code, but a user could
    // write that and we should handle it).
    firOpBuilder.setInsertionPoint(term);
    if (privatize) {
      // DataSharingProcessor::processStep2() may create operations before/after
      // the one passed as argument. We need to treat loop wrappers and their
      // nested loop as a unit, so we need to pass the top level wrapper (if
      // present). Otherwise, these operations will be inserted within a
      // wrapper region.
      mlir::Operation *privatizationTopLevelOp = &op;
      if (auto loopNest = llvm::dyn_cast<mlir::omp::LoopNestOp>(op)) {
        llvm::SmallVector<mlir::omp::LoopWrapperInterface> wrappers;
        loopNest.gatherWrappers(wrappers);
        if (!wrappers.empty())
          privatizationTopLevelOp = &*wrappers.back();
      }

      if (!info.dsp) {
        assert(tempDsp.has_value());
        tempDsp->processStep2(privatizationTopLevelOp, isLoop);
      } else {
        if (isLoop && regionArgs.size() > 0)
          info.dsp->setLoopIV(info.converter.getSymbolAddress(*regionArgs[0]));
        info.dsp->processStep2(privatizationTopLevelOp, isLoop);
      }
    }
  }

  firOpBuilder.setInsertionPointAfter(marker);
  marker->erase();
}

static void genBodyOfTargetDataOp(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::omp::TargetDataOp &dataOp, llvm::ArrayRef<mlir::Type> useDeviceTypes,
    llvm::ArrayRef<mlir::Location> useDeviceLocs,
    llvm::ArrayRef<const semantics::Symbol *> useDeviceSymbols,
    const mlir::Location &currentLocation, const ConstructQueue &queue,
    ConstructQueue::iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = dataOp.getRegion();

  firOpBuilder.createBlock(&region, {}, useDeviceTypes, useDeviceLocs);

  for (auto [argIndex, argSymbol] : llvm::enumerate(useDeviceSymbols)) {
    const mlir::BlockArgument &arg = region.front().getArgument(argIndex);
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*argSymbol);
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(arg.getType())) {
      if (fir::isa_builtin_cptr_type(refType.getElementType())) {
        converter.bindSymbol(*argSymbol, arg);
      } else {
        // Avoid capture of a reference to a structured binding.
        const semantics::Symbol *sym = argSymbol;
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
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Set the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());

  if (ConstructQueue::iterator next = std::next(item); next != queue.end()) {
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   next);
  } else {
    genNestedEvaluations(converter, eval);
  }
}

// This functions creates a block for the body of the targetOp's region. It adds
// all the symbols present in mapSymbols as block arguments to this block.
static void
genBodyOfTargetOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                  semantics::SemanticsContext &semaCtx,
                  lower::pft::Evaluation &eval, mlir::omp::TargetOp &targetOp,
                  llvm::ArrayRef<const semantics::Symbol *> mapSyms,
                  llvm::ArrayRef<mlir::Location> mapSymLocs,
                  llvm::ArrayRef<mlir::Type> mapSymTypes,
                  const mlir::Location &currentLocation,
                  const ConstructQueue &queue, ConstructQueue::iterator item) {
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
  for (auto [argIndex, argSymbol] : llvm::enumerate(mapSyms)) {
    const mlir::BlockArgument &arg = region.getArgument(argIndex);
    // Avoid capture of a reference to a structured binding.
    const semantics::Symbol *sym = argSymbol;
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
            firOpBuilder, copyVal.getLoc(), copyVal,
            /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
            /*members=*/llvm::SmallVector<mlir::Value>{},
            /*membersIndex=*/mlir::DenseIntElementsAttr{},
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
    lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp, mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Create the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());

  if (ConstructQueue::iterator next = std::next(item); next != queue.end()) {
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   next);
  } else {
    genNestedEvaluations(converter, eval);
  }
}

template <typename OpTy, typename... Args>
static OpTy genOpWithBody(const OpWithBodyGenInfo &info,
                          const ConstructQueue &queue,
                          ConstructQueue::iterator item, Args &&...args) {
  auto op = info.converter.getFirOpBuilder().create<OpTy>(
      info.loc, std::forward<Args>(args)...);
  createBodyOfOp(*op, info, queue, item);
  return op;
}

//===----------------------------------------------------------------------===//
// Code generation functions for clauses
//===----------------------------------------------------------------------===//

static void genCriticalDeclareClauses(lower::AbstractConverter &converter,
                                      semantics::SemanticsContext &semaCtx,
                                      const List<Clause> &clauses,
                                      mlir::Location loc,
                                      mlir::omp::CriticalClauseOps &clauseOps,
                                      llvm::StringRef name) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processHint(clauseOps);
  clauseOps.nameAttr =
      mlir::StringAttr::get(converter.getFirOpBuilder().getContext(), name);
}

static void genFlushClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            const ObjectList &objects,
                            const List<Clause> &clauses, mlir::Location loc,
                            llvm::SmallVectorImpl<mlir::Value> &operandRange) {
  if (!objects.empty())
    genObjectList(objects, converter, operandRange);

  if (!clauses.empty())
    TODO(converter.getCurrentLocation(), "Handle OmpMemoryOrderClause");
}

static void
genLoopNestClauses(lower::AbstractConverter &converter,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval, const List<Clause> &clauses,
                   mlir::Location loc, mlir::omp::LoopNestClauseOps &clauseOps,
                   llvm::SmallVectorImpl<const semantics::Symbol *> &iv) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processCollapse(loc, eval, clauseOps, iv);
  clauseOps.loopInclusiveAttr = converter.getFirOpBuilder().getUnitAttr();
}

static void
genOrderedRegionClauses(lower::AbstractConverter &converter,
                        semantics::SemanticsContext &semaCtx,
                        const List<Clause> &clauses, mlir::Location loc,
                        mlir::omp::OrderedRegionClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<clause::Simd>(loc, llvm::omp::Directive::OMPD_ordered);
}

static void genParallelClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, bool processReduction,
    mlir::omp::ParallelClauseOps &clauseOps,
    llvm::SmallVectorImpl<mlir::Type> &reductionTypes,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processIf(llvm::omp::Directive::OMPD_parallel, clauseOps);
  cp.processNumThreads(stmtCtx, clauseOps);
  cp.processProcBind(clauseOps);

  if (processReduction) {
    cp.processReduction(loc, clauseOps, &reductionTypes, &reductionSyms);
  }
}

static void genSectionsClauses(lower::AbstractConverter &converter,
                               semantics::SemanticsContext &semaCtx,
                               const List<Clause> &clauses, mlir::Location loc,
                               mlir::omp::SectionsClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processSectionsReduction(loc, clauseOps);
  cp.processNowait(clauseOps);
  // TODO Support delayed privatization.
}

static void genSimdClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::SimdClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processIf(llvm::omp::Directive::OMPD_simd, clauseOps);
  cp.processReduction(loc, clauseOps);
  cp.processSafelen(clauseOps);
  cp.processSimdlen(clauseOps);
  // TODO Support delayed privatization.

  cp.processTODO<clause::Aligned, clause::Allocate, clause::Linear,
                 clause::Nontemporal, clause::Order>(
      loc, llvm::omp::Directive::OMPD_simd);
}

static void genSingleClauses(lower::AbstractConverter &converter,
                             semantics::SemanticsContext &semaCtx,
                             const List<Clause> &clauses, mlir::Location loc,
                             mlir::omp::SingleClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processCopyprivate(loc, clauseOps);
  cp.processNowait(clauseOps);
  // TODO Support delayed privatization.
}

static void genTargetClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, bool processHostOnlyClauses, bool processReduction,
    mlir::omp::TargetClauseOps &clauseOps,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
    llvm::SmallVectorImpl<mlir::Location> &mapLocs,
    llvm::SmallVectorImpl<mlir::Type> &mapTypes,
    llvm::SmallVectorImpl<const semantics::Symbol *> &deviceAddrSyms,
    llvm::SmallVectorImpl<mlir::Location> &deviceAddrLocs,
    llvm::SmallVectorImpl<mlir::Type> &deviceAddrTypes,
    llvm::SmallVectorImpl<const semantics::Symbol *> &devicePtrSyms,
    llvm::SmallVectorImpl<mlir::Location> &devicePtrLocs,
    llvm::SmallVectorImpl<mlir::Type> &devicePtrTypes) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processHasDeviceAddr(clauseOps, deviceAddrTypes, deviceAddrLocs,
                          deviceAddrSyms);
  cp.processIf(llvm::omp::Directive::OMPD_target, clauseOps);
  cp.processIsDevicePtr(clauseOps, devicePtrTypes, devicePtrLocs,
                        devicePtrSyms);
  cp.processMap(loc, stmtCtx, clauseOps, &mapSyms, &mapLocs, &mapTypes);
  cp.processThreadLimit(stmtCtx, clauseOps);
  // TODO Support delayed privatization.

  if (processHostOnlyClauses)
    cp.processNowait(clauseOps);

  cp.processTODO<clause::Allocate, clause::Defaultmap, clause::Firstprivate,
                 clause::InReduction, clause::Private, clause::Reduction,
                 clause::UsesAllocators>(loc,
                                         llvm::omp::Directive::OMPD_target);
}

static void genTargetDataClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::TargetDataClauseOps &clauseOps,
    llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
    llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
    llvm::SmallVectorImpl<const semantics::Symbol *> &useDeviceSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_target_data, clauseOps);
  cp.processMap(loc, stmtCtx, clauseOps);
  cp.processUseDeviceAddr(clauseOps, useDeviceTypes, useDeviceLocs,
                          useDeviceSyms);
  cp.processUseDevicePtr(clauseOps, useDeviceTypes, useDeviceLocs,
                         useDeviceSyms);

  // This function implements the deprecated functionality of use_device_ptr
  // that allows users to provide non-CPTR arguments to it with the caveat
  // that the compiler will treat them as use_device_addr. A lot of legacy
  // code may still depend on this functionality, so we should support it
  // in some manner. We do so currently by simply shifting non-cptr operands
  // from the use_device_ptr list into the front of the use_device_addr list
  // whilst maintaining the ordering of useDeviceLocs, useDeviceSyms and
  // useDeviceTypes to use_device_ptr/use_device_addr input for BlockArg
  // ordering.
  // TODO: Perhaps create a user provideable compiler option that will
  // re-introduce a hard-error rather than a warning in these cases.
  promoteNonCPtrUseDevicePtrArgsToUseDeviceAddr(clauseOps, useDeviceTypes,
                                                useDeviceLocs, useDeviceSyms);
}

static void genTargetEnterExitUpdateDataClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, llvm::omp::Directive directive,
    mlir::omp::TargetEnterExitUpdateDataClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(directive, clauseOps);
  cp.processNowait(clauseOps);

  if (directive == llvm::omp::Directive::OMPD_target_update) {
    cp.processMotionClauses<clause::To>(stmtCtx, clauseOps);
    cp.processMotionClauses<clause::From>(stmtCtx, clauseOps);
  } else {
    cp.processMap(loc, stmtCtx, clauseOps);
  }
}

static void genTaskClauses(lower::AbstractConverter &converter,
                           semantics::SemanticsContext &semaCtx,
                           lower::StatementContext &stmtCtx,
                           const List<Clause> &clauses, mlir::Location loc,
                           mlir::omp::TaskClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processDepend(clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(llvm::omp::Directive::OMPD_task, clauseOps);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processUntied(clauseOps);
  // TODO Support delayed privatization.

  cp.processTODO<clause::Affinity, clause::Detach, clause::InReduction>(
      loc, llvm::omp::Directive::OMPD_task);
}

static void genTaskgroupClauses(lower::AbstractConverter &converter,
                                semantics::SemanticsContext &semaCtx,
                                const List<Clause> &clauses, mlir::Location loc,
                                mlir::omp::TaskgroupClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTODO<clause::TaskReduction>(loc,
                                        llvm::omp::Directive::OMPD_taskgroup);
}

static void genTaskwaitClauses(lower::AbstractConverter &converter,
                               semantics::SemanticsContext &semaCtx,
                               const List<Clause> &clauses, mlir::Location loc,
                               mlir::omp::TaskwaitClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<clause::Depend, clause::Nowait>(
      loc, llvm::omp::Directive::OMPD_taskwait);
}

static void genTeamsClauses(lower::AbstractConverter &converter,
                            semantics::SemanticsContext &semaCtx,
                            lower::StatementContext &stmtCtx,
                            const List<Clause> &clauses, mlir::Location loc,
                            mlir::omp::TeamsClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processIf(llvm::omp::Directive::OMPD_teams, clauseOps);
  cp.processNumTeams(stmtCtx, clauseOps);
  cp.processThreadLimit(stmtCtx, clauseOps);
  // TODO Support delayed privatization.

  cp.processTODO<clause::Reduction>(loc, llvm::omp::Directive::OMPD_teams);
}

static void genWsloopClauses(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::StatementContext &stmtCtx, const List<Clause> &clauses,
    mlir::Location loc, mlir::omp::WsloopClauseOps &clauseOps,
    llvm::SmallVectorImpl<mlir::Type> &reductionTypes,
    llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSyms) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processNowait(clauseOps);
  cp.processOrdered(clauseOps);
  cp.processReduction(loc, clauseOps, &reductionTypes, &reductionSyms);
  cp.processSchedule(stmtCtx, clauseOps);
  // TODO Support delayed privatization.

  cp.processTODO<clause::Allocate, clause::Linear, clause::Order>(
      loc, llvm::omp::Directive::OMPD_do);
}

//===----------------------------------------------------------------------===//
// Code generation functions for leaf constructs
//===----------------------------------------------------------------------===//

static mlir::omp::BarrierOp
genBarrierOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
             semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
             mlir::Location loc, const ConstructQueue &queue,
             ConstructQueue::iterator item) {
  return converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(loc);
}

static mlir::omp::CriticalOp
genCriticalOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::iterator item,
              const std::optional<parser::Name> &name) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::FlatSymbolRefAttr nameAttr;

  if (name) {
    std::string nameStr = name->ToString();
    mlir::ModuleOp mod = firOpBuilder.getModule();
    auto global = mod.lookupSymbol<mlir::omp::CriticalDeclareOp>(nameStr);
    if (!global) {
      mlir::omp::CriticalClauseOps clauseOps;
      genCriticalDeclareClauses(converter, semaCtx, item->clauses, loc,
                                clauseOps, nameStr);

      mlir::OpBuilder modBuilder(mod.getBodyRegion());
      global = modBuilder.create<mlir::omp::CriticalDeclareOp>(loc, clauseOps);
    }
    nameAttr = mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(),
                                            global.getSymName());
  }

  return genOpWithBody<mlir::omp::CriticalOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_critical),
      queue, item, nameAttr);
}

static mlir::omp::DistributeOp
genDistributeOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                semantics::SemanticsContext &semaCtx,
                lower::pft::Evaluation &eval, mlir::Location loc,
                const ConstructQueue &queue, ConstructQueue::iterator item) {
  TODO(loc, "Distribute construct");
  return nullptr;
}

static mlir::omp::FlushOp
genFlushOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ObjectList &objects,
           const ConstructQueue &queue, ConstructQueue::iterator item) {
  llvm::SmallVector<mlir::Value> operandRange;
  genFlushClauses(converter, semaCtx, objects, item->clauses, loc,
                  operandRange);

  return converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
      converter.getCurrentLocation(), operandRange);
}

static mlir::omp::MasterOp
genMasterOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::iterator item) {
  return genOpWithBody<mlir::omp::MasterOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_master),
      queue, item);
}

static mlir::omp::OrderedOp
genOrderedOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
             semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
             mlir::Location loc, const ConstructQueue &queue,
             ConstructQueue::iterator item) {
  TODO(loc, "OMPD_ordered");
  return nullptr;
}

static mlir::omp::OrderedRegionOp
genOrderedRegionOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval, mlir::Location loc,
                   const ConstructQueue &queue, ConstructQueue::iterator item) {
  mlir::omp::OrderedRegionClauseOps clauseOps;
  genOrderedRegionClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::OrderedRegionOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_ordered),
      queue, item, clauseOps);
}

static mlir::omp::ParallelOp
genParallelOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::iterator item,
              bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;
  mlir::omp::ParallelClauseOps clauseOps;
  llvm::SmallVector<const semantics::Symbol *> privateSyms;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<const semantics::Symbol *> reductionSyms;
  genParallelClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                     /*processReduction=*/!outerCombined, clauseOps,
                     reductionTypes, reductionSyms);

  auto reductionCallback = [&](mlir::Operation *op) {
    genReductionVars(op, converter, loc, reductionSyms, reductionTypes);
    return reductionSyms;
  };

  OpWithBodyGenInfo genInfo =
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_parallel)
          .setOuterCombined(outerCombined)
          .setClauses(&item->clauses)
          .setReductions(&reductionSyms, &reductionTypes)
          .setGenRegionEntryCb(reductionCallback);

  if (!enableDelayedPrivatization)
    return genOpWithBody<mlir::omp::ParallelOp>(genInfo, queue, item,
                                                clauseOps);

  bool privatize = !outerCombined;
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue),
                           /*useDelayedPrivatization=*/true, &symTable);

  if (privatize)
    dsp.processStep1(&clauseOps, &privateSyms);

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    auto parallelOp = llvm::cast<mlir::omp::ParallelOp>(op);

    llvm::SmallVector<mlir::Location> reductionLocs(
        clauseOps.reductionVars.size(), loc);

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

    firOpBuilder.createBlock(&region, /*insertPt=*/{}, privateVarTypes,
                             privateVarLocs);

    llvm::SmallVector<const semantics::Symbol *> allSymbols = reductionSyms;
    allSymbols.append(privateSyms);
    for (auto [arg, prv] : llvm::zip_equal(allSymbols, region.getArguments())) {
      fir::ExtendedValue hostExV = converter.getSymbolExtendedValue(*arg);
      converter.bindSymbol(*arg, hlfir::translateToExtendedValue(
                                     loc, firOpBuilder, hlfir::Entity{prv},
                                     /*contiguousHint=*/
                                     evaluate::IsSimplyContiguous(
                                         *arg, converter.getFoldingContext()))
                                     .first);
    }

    return allSymbols;
  };

  genInfo.setGenRegionEntryCb(genRegionEntryCB).setDataSharingProcessor(&dsp);
  return genOpWithBody<mlir::omp::ParallelOp>(genInfo, queue, item, clauseOps);
}

static mlir::omp::SectionOp
genSectionOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
             semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
             mlir::Location loc, const ConstructQueue &queue,
             ConstructQueue::iterator item) {
  // Currently only private/firstprivate clause is handled, and
  // all privatization is done within `omp.section` operations.
  return genOpWithBody<mlir::omp::SectionOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_section)
          .setClauses(&item->clauses),
      queue, item);
}

static mlir::omp::SectionsOp
genSectionsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::iterator item) {
  mlir::omp::SectionsClauseOps clauseOps;
  genSectionsClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  auto &builder = converter.getFirOpBuilder();

  // Insert privatizations before SECTIONS
  symTable.pushScope();
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue));
  dsp.processStep1();

  List<Clause> nonDsaClauses;
  List<const clause::Lastprivate *> lastprivates;

  for (const Clause &clause : item->clauses) {
    if (clause.id == llvm::omp::Clause::OMPC_lastprivate) {
      lastprivates.push_back(&std::get<clause::Lastprivate>(clause.u));
    } else {
      switch (clause.id) {
      case llvm::omp::Clause::OMPC_firstprivate:
      case llvm::omp::Clause::OMPC_private:
      case llvm::omp::Clause::OMPC_shared:
        break;
      default:
        nonDsaClauses.push_back(clause);
      }
    }
  }

  // SECTIONS construct.
  mlir::omp::SectionsOp sectionsOp = genOpWithBody<mlir::omp::SectionsOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_sections)
          .setClauses(&nonDsaClauses),
      queue, item, clauseOps);

  if (!lastprivates.empty()) {
    mlir::Region &sectionsBody = sectionsOp.getRegion();
    assert(sectionsBody.hasOneBlock());
    mlir::Block &body = sectionsBody.front();

    auto lastSectionOp = llvm::find_if(
        llvm::reverse(body.getOperations()), [](const mlir::Operation &op) {
          return llvm::isa<mlir::omp::SectionOp>(op);
        });
    assert(lastSectionOp != body.rend());

    for (const clause::Lastprivate *lastp : lastprivates) {
      builder.setInsertionPoint(
          lastSectionOp->getRegion(0).back().getTerminator());
      mlir::OpBuilder::InsertPoint insp = builder.saveInsertionPoint();
      const auto &objList = std::get<ObjectList>(lastp->t);
      for (const Object &object : objList) {
        semantics::Symbol *sym = object.id();
        converter.copyHostAssociateVar(*sym, &insp);
      }
    }
  }

  // Perform DataSharingProcessor's step2 out of SECTIONS
  builder.setInsertionPointAfter(sectionsOp.getOperation());
  dsp.processStep2(sectionsOp, false);
  // Emit implicit barrier to synchronize threads and avoid data
  // races on post-update of lastprivate variables when `nowait`
  // clause is present.
  if (clauseOps.nowaitAttr && !lastprivates.empty())
    builder.create<mlir::omp::BarrierOp>(loc);

  symTable.popScope();
  return sectionsOp;
}

static mlir::omp::SimdOp
genSimdOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
          ConstructQueue::iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  symTable.pushScope();
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue));
  dsp.processStep1();

  lower::StatementContext stmtCtx;
  mlir::omp::LoopNestClauseOps loopClauseOps;
  mlir::omp::SimdClauseOps simdClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopClauseOps, iv);
  genSimdClauses(converter, semaCtx, item->clauses, loc, simdClauseOps);

  // Create omp.simd wrapper.
  auto simdOp = firOpBuilder.create<mlir::omp::SimdOp>(loc, simdClauseOps);

  // TODO: Add reduction-related arguments to the wrapper's entry block.
  firOpBuilder.createBlock(&simdOp.getRegion());
  firOpBuilder.setInsertionPoint(
      lower::genOpenMPTerminator(firOpBuilder, simdOp, loc));

  // Create nested omp.loop_nest and fill body with loop contents.
  auto loopOp = firOpBuilder.create<mlir::omp::LoopNestOp>(loc, loopClauseOps);

  auto *nestedEval =
      getCollapsedLoopEval(eval, getCollapseValue(item->clauses));

  auto ivCallback = [&](mlir::Operation *op) {
    genLoopVars(op, converter, loc, iv);
    return iv;
  };

  createBodyOfOp(*loopOp,
                 OpWithBodyGenInfo(converter, symTable, semaCtx, loc,
                                   *nestedEval, llvm::omp::Directive::OMPD_simd)
                     .setClauses(&item->clauses)
                     .setDataSharingProcessor(&dsp)
                     .setGenRegionEntryCb(ivCallback),
                 queue, item);

  symTable.popScope();
  return simdOp;
}

static mlir::omp::SingleOp
genSingleOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::iterator item) {
  mlir::omp::SingleClauseOps clauseOps;
  genSingleClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::SingleOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_single)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TargetOp
genTargetOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::iterator item, bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;

  bool processHostOnlyClauses =
      !llvm::cast<mlir::omp::OffloadModuleInterface>(*converter.getModuleOp())
           .getIsTargetDevice();

  mlir::omp::TargetClauseOps clauseOps;
  llvm::SmallVector<const semantics::Symbol *> mapSyms, devicePtrSyms,
      deviceAddrSyms;
  llvm::SmallVector<mlir::Location> mapLocs, devicePtrLocs, deviceAddrLocs;
  llvm::SmallVector<mlir::Type> mapTypes, devicePtrTypes, deviceAddrTypes;
  genTargetClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                   processHostOnlyClauses, /*processReduction=*/outerCombined,
                   clauseOps, mapSyms, mapLocs, mapTypes, deviceAddrSyms,
                   deviceAddrLocs, deviceAddrTypes, devicePtrSyms,
                   devicePtrLocs, devicePtrTypes);

  // 5.8.1 Implicit Data-Mapping Attribute Rules
  // The following code follows the implicit data-mapping rules to map all the
  // symbols used inside the region that have not been explicitly mapped using
  // the map clause.
  auto captureImplicitMap = [&](const semantics::Symbol &sym) {
    if (llvm::find(mapSyms, &sym) == mapSyms.end()) {
      mlir::Value baseOp = converter.getSymbolAddress(sym);
      if (!baseOp)
        if (const auto *details =
                sym.template detailsIf<semantics::HostAssocDetails>()) {
          baseOp = converter.getSymbolAddress(details->symbol());
          converter.copySymbolBinding(details->symbol(), sym);
        }

      if (baseOp) {
        llvm::SmallVector<mlir::Value> bounds;
        std::stringstream name;
        fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
        name << sym.name().ToString();

        lower::AddrAndBoundsInfo info = getDataOperandBaseAddr(
            converter, firOpBuilder, sym, converter.getCurrentLocation());
        if (mlir::isa<fir::BaseBoxType>(
                fir::unwrapRefType(info.addr.getType())))
          bounds = lower::genBoundsOpsFromBox<mlir::omp::MapBoundsOp,
                                              mlir::omp::MapBoundsType>(
              firOpBuilder, converter.getCurrentLocation(), converter, dataExv,
              info);
        if (mlir::isa<fir::SequenceType>(
                fir::unwrapRefType(info.addr.getType()))) {
          bool dataExvIsAssumedSize =
              semantics::IsAssumedSizeArray(sym.GetUltimate());
          bounds = lower::genBaseBoundsOps<mlir::omp::MapBoundsOp,
                                           mlir::omp::MapBoundsType>(
              firOpBuilder, converter.getCurrentLocation(), converter, dataExv,
              dataExvIsAssumedSize);
        }

        llvm::omp::OpenMPOffloadMappingFlags mapFlag =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
        mlir::omp::VariableCaptureKind captureKind =
            mlir::omp::VariableCaptureKind::ByRef;

        mlir::Type eleType = baseOp.getType();
        if (auto refType = mlir::dyn_cast<fir::ReferenceType>(baseOp.getType()))
          eleType = refType.getElementType();

        // If a variable is specified in declare target link and if device
        // type is not specified as `nohost`, it needs to be mapped tofrom
        mlir::ModuleOp mod = firOpBuilder.getModule();
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
        auto location =
            mlir::NameLoc::get(mlir::StringAttr::get(firOpBuilder.getContext(),
                                                     sym.name().ToString()),
                               baseOp.getLoc());
        mlir::Value mapOp = createMapInfoOp(
            firOpBuilder, location, baseOp, /*varPtrPtr=*/mlir::Value{},
            name.str(), bounds, /*members=*/{},
            /*membersIndex=*/mlir::DenseIntElementsAttr{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                mapFlag),
            captureKind, baseOp.getType());

        clauseOps.mapVars.push_back(mapOp);
        mapSyms.push_back(&sym);
        mapLocs.push_back(baseOp.getLoc());
        mapTypes.push_back(baseOp.getType());
      }
    }
  };
  lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = firOpBuilder.create<mlir::omp::TargetOp>(loc, clauseOps);
  genBodyOfTargetOp(converter, symTable, semaCtx, eval, targetOp, mapSyms,
                    mapLocs, mapTypes, loc, queue, item);
  return targetOp;
}

static mlir::omp::TargetDataOp
genTargetDataOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
                semantics::SemanticsContext &semaCtx,
                lower::pft::Evaluation &eval, mlir::Location loc,
                const ConstructQueue &queue, ConstructQueue::iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::TargetDataClauseOps clauseOps;
  llvm::SmallVector<mlir::Type> useDeviceTypes;
  llvm::SmallVector<mlir::Location> useDeviceLocs;
  llvm::SmallVector<const semantics::Symbol *> useDeviceSyms;
  genTargetDataClauses(converter, semaCtx, stmtCtx, item->clauses, loc,
                       clauseOps, useDeviceTypes, useDeviceLocs, useDeviceSyms);

  auto targetDataOp =
      converter.getFirOpBuilder().create<mlir::omp::TargetDataOp>(loc,
                                                                  clauseOps);
  genBodyOfTargetDataOp(converter, symTable, semaCtx, eval, targetDataOp,
                        useDeviceTypes, useDeviceLocs, useDeviceSyms, loc,
                        queue, item);
  return targetDataOp;
}

template <typename OpTy>
static OpTy genTargetEnterExitUpdateDataOp(lower::AbstractConverter &converter,
                                           lower::SymMap &symTable,
                                           semantics::SemanticsContext &semaCtx,
                                           mlir::Location loc,
                                           const ConstructQueue &queue,
                                           ConstructQueue::iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  lower::StatementContext stmtCtx;

  // GCC 9.3.0 emits a (probably) bogus warning about an unused variable.
  [[maybe_unused]] llvm::omp::Directive directive;
  if constexpr (std::is_same_v<OpTy, mlir::omp::TargetEnterDataOp>) {
    directive = llvm::omp::Directive::OMPD_target_enter_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetExitDataOp>) {
    directive = llvm::omp::Directive::OMPD_target_exit_data;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::TargetUpdateOp>) {
    directive = llvm::omp::Directive::OMPD_target_update;
  } else {
    llvm_unreachable("Unexpected TARGET DATA construct");
  }

  mlir::omp::TargetEnterExitUpdateDataClauseOps clauseOps;
  genTargetEnterExitUpdateDataClauses(converter, semaCtx, stmtCtx,
                                      item->clauses, loc, directive, clauseOps);

  return firOpBuilder.create<OpTy>(loc, clauseOps);
}

static mlir::omp::TaskOp
genTaskOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
          semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
          mlir::Location loc, const ConstructQueue &queue,
          ConstructQueue::iterator item) {
  lower::StatementContext stmtCtx;
  mlir::omp::TaskClauseOps clauseOps;
  genTaskClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::TaskOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_task)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TaskgroupOp
genTaskgroupOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue, ConstructQueue::iterator item) {
  mlir::omp::TaskgroupClauseOps clauseOps;
  genTaskgroupClauses(converter, semaCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::TaskgroupOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_taskgroup)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::TaskloopOp
genTaskloopOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::iterator item) {
  TODO(loc, "Taskloop construct");
}

static mlir::omp::TaskwaitOp
genTaskwaitOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
              semantics::SemanticsContext &semaCtx,
              lower::pft::Evaluation &eval, mlir::Location loc,
              const ConstructQueue &queue, ConstructQueue::iterator item) {
  mlir::omp::TaskwaitClauseOps clauseOps;
  genTaskwaitClauses(converter, semaCtx, item->clauses, loc, clauseOps);
  return converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(loc,
                                                                   clauseOps);
}

static mlir::omp::TaskyieldOp
genTaskyieldOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
               semantics::SemanticsContext &semaCtx,
               lower::pft::Evaluation &eval, mlir::Location loc,
               const ConstructQueue &queue, ConstructQueue::iterator item) {
  return converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(loc);
}

static mlir::omp::TeamsOp
genTeamsOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
           semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
           mlir::Location loc, const ConstructQueue &queue,
           ConstructQueue::iterator item, bool outerCombined = false) {
  lower::StatementContext stmtCtx;
  mlir::omp::TeamsClauseOps clauseOps;
  genTeamsClauses(converter, semaCtx, stmtCtx, item->clauses, loc, clauseOps);

  return genOpWithBody<mlir::omp::TeamsOp>(
      OpWithBodyGenInfo(converter, symTable, semaCtx, loc, eval,
                        llvm::omp::Directive::OMPD_teams)
          .setOuterCombined(outerCombined)
          .setClauses(&item->clauses),
      queue, item, clauseOps);
}

static mlir::omp::WsloopOp
genWsloopOp(lower::AbstractConverter &converter, lower::SymMap &symTable,
            semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
            mlir::Location loc, const ConstructQueue &queue,
            ConstructQueue::iterator item) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  symTable.pushScope();
  DataSharingProcessor dsp(converter, semaCtx, item->clauses, eval,
                           lower::omp::isLastItemInQueue(item, queue));
  dsp.processStep1();

  lower::StatementContext stmtCtx;
  mlir::omp::LoopNestClauseOps loopClauseOps;
  mlir::omp::WsloopClauseOps wsClauseOps;
  llvm::SmallVector<const semantics::Symbol *> iv;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<const semantics::Symbol *> reductionSyms;
  genLoopNestClauses(converter, semaCtx, eval, item->clauses, loc,
                     loopClauseOps, iv);
  genWsloopClauses(converter, semaCtx, stmtCtx, item->clauses, loc, wsClauseOps,
                   reductionTypes, reductionSyms);

  // Create omp.wsloop wrapper and populate entry block arguments with reduction
  // variables.
  auto wsloopOp = firOpBuilder.create<mlir::omp::WsloopOp>(loc, wsClauseOps);
  llvm::SmallVector<mlir::Location> reductionLocs(reductionSyms.size(), loc);
  mlir::Block *wsloopEntryBlock = firOpBuilder.createBlock(
      &wsloopOp.getRegion(), {}, reductionTypes, reductionLocs);
  firOpBuilder.setInsertionPoint(
      lower::genOpenMPTerminator(firOpBuilder, wsloopOp, loc));

  // Create nested omp.loop_nest and fill body with loop contents.
  auto loopOp = firOpBuilder.create<mlir::omp::LoopNestOp>(loc, loopClauseOps);

  auto *nestedEval =
      getCollapsedLoopEval(eval, getCollapseValue(item->clauses));

  auto ivCallback = [&](mlir::Operation *op) {
    genLoopVars(op, converter, loc, iv, reductionSyms,
                wsloopEntryBlock->getArguments());
    return iv;
  };

  createBodyOfOp(*loopOp,
                 OpWithBodyGenInfo(converter, symTable, semaCtx, loc,
                                   *nestedEval, llvm::omp::Directive::OMPD_do)
                     .setClauses(&item->clauses)
                     .setDataSharingProcessor(&dsp)
                     .setReductions(&reductionSyms, &reductionTypes)
                     .setGenRegionEntryCb(ivCallback),
                 queue, item);
  symTable.popScope();
  return wsloopOp;
}

//===----------------------------------------------------------------------===//
// Code generation functions for composite constructs
//===----------------------------------------------------------------------===//

static void genCompositeDistributeParallelDo(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::iterator item) {
  TODO(loc, "Composite DISTRIBUTE PARALLEL DO");
}

static void genCompositeDistributeParallelDoSimd(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    mlir::Location loc, const ConstructQueue &queue,
    ConstructQueue::iterator item) {
  TODO(loc, "Composite DISTRIBUTE PARALLEL DO SIMD");
}

static void genCompositeDistributeSimd(lower::AbstractConverter &converter,
                                       lower::SymMap &symTable,
                                       semantics::SemanticsContext &semaCtx,
                                       lower::pft::Evaluation &eval,
                                       mlir::Location loc,
                                       const ConstructQueue &queue,
                                       ConstructQueue::iterator item) {
  TODO(loc, "Composite DISTRIBUTE SIMD");
}

static void genCompositeDoSimd(lower::AbstractConverter &converter,
                               lower::SymMap &symTable,
                               semantics::SemanticsContext &semaCtx,
                               lower::pft::Evaluation &eval, mlir::Location loc,
                               const ConstructQueue &queue,
                               ConstructQueue::iterator item) {
  ClauseProcessor cp(converter, semaCtx, item->clauses);
  cp.processTODO<clause::Aligned, clause::Allocate, clause::Linear,
                 clause::Order, clause::Safelen, clause::Simdlen>(
      loc, llvm::omp::OMPD_do_simd);
  // TODO: Add support for vectorization - add vectorization hints inside loop
  // body.
  // OpenMP standard does not specify the length of vector instructions.
  // Currently we safely assume that for !$omp do simd pragma the SIMD length
  // is equal to 1 (i.e. we generate standard workshare loop).
  // When support for vectorization is enabled, then we need to add handling of
  // if clause. Currently if clause can be skipped because we always assume
  // SIMD length = 1.
  genWsloopOp(converter, symTable, semaCtx, eval, loc, queue, item);
}

static void genCompositeTaskloopSimd(lower::AbstractConverter &converter,
                                     lower::SymMap &symTable,
                                     semantics::SemanticsContext &semaCtx,
                                     lower::pft::Evaluation &eval,
                                     mlir::Location loc,
                                     const ConstructQueue &queue,
                                     ConstructQueue::iterator item) {
  TODO(loc, "Composite TASKLOOP SIMD");
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static void genOMPDispatch(lower::AbstractConverter &converter,
                           lower::SymMap &symTable,
                           semantics::SemanticsContext &semaCtx,
                           lower::pft::Evaluation &eval, mlir::Location loc,
                           const ConstructQueue &queue,
                           ConstructQueue::iterator item) {
  assert(item != queue.end());

  switch (llvm::omp::Directive dir = item->id) {
  case llvm::omp::Directive::OMPD_barrier:
    genBarrierOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_distribute:
    genDistributeOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_do:
    genWsloopOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_loop:
  case llvm::omp::Directive::OMPD_masked:
    TODO(loc, "Unhandled directive " + llvm::omp::getOpenMPDirectiveName(dir));
    break;
  case llvm::omp::Directive::OMPD_master:
    genMasterOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    // Block-associated "ordered" construct.
    genOrderedRegionOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    genParallelOp(converter, symTable, semaCtx, eval, loc, queue, item,
                  /*outerCombined=*/false);
    break;
  case llvm::omp::Directive::OMPD_section:
    genSectionOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_sections:
    genSectionsOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_simd:
    genSimdOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_single:
    genSingleOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target:
    genTargetOp(converter, symTable, semaCtx, eval, loc, queue, item,
                /*outerCombined=*/false);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genTargetDataOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetEnterDataOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetExitDataOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    genTargetEnterExitUpdateDataOp<mlir::omp::TargetUpdateOp>(
        converter, symTable, semaCtx, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_task:
    genTaskOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    genTaskgroupOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskloop:
    genTaskloopOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    genTaskwaitOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    genTaskyieldOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_teams:
    genTeamsOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll:
    TODO(loc, "Unhandled loop directive (" +
                  llvm::omp::getOpenMPDirectiveName(dir) + ")");
  // case llvm::omp::Directive::OMPD_workdistribute:
  case llvm::omp::Directive::OMPD_workshare:
    // FIXME: Workshare is not a commonly used OpenMP construct, an
    // implementation for this feature will come later. For the codes
    // that use this construct, add a single construct for now.
    genSingleOp(converter, symTable, semaCtx, eval, loc, queue, item);
    break;

  // Composite constructs
  case llvm::omp::Directive::OMPD_distribute_parallel_do:
    genCompositeDistributeParallelDo(converter, symTable, semaCtx, eval, loc,
                                     queue, item);
    break;
  case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
    genCompositeDistributeParallelDoSimd(converter, symTable, semaCtx, eval,
                                         loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_distribute_simd:
    genCompositeDistributeSimd(converter, symTable, semaCtx, eval, loc, queue,
                               item);
    break;
  case llvm::omp::Directive::OMPD_do_simd:
    genCompositeDoSimd(converter, symTable, semaCtx, eval, loc, queue, item);
    break;
  case llvm::omp::Directive::OMPD_taskloop_simd:
    genCompositeTaskloopSimd(converter, symTable, semaCtx, eval, loc, queue,
                             item);
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// OpenMPDeclarativeConstruct visitors
//===----------------------------------------------------------------------===//

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclarativeAllocate &declarativeAllocate) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
}

static void genOMP(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPDeclareReductionConstruct &declareReductionConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclareReductionConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareSimdConstruct &declareSimdConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPDeclareTargetConstruct &declareTargetConstruct) {
  mlir::omp::DeclareTargetClauseOps clauseOps;
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  getDeclareTargetInfo(converter, semaCtx, eval, declareTargetConstruct,
                       clauseOps, symbolAndClause);

  for (const DeclareTargetCapturePair &symClause : symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(
        converter.mangleName(std::get<const semantics::Symbol &>(symClause)));

    // Some symbols are deferred until later in the module, these are handled
    // upon finalization of the module for OpenMP inside of Bridge, so we simply
    // skip for now.
    if (!op)
      continue;

    markDeclareTarget(
        op, converter,
        std::get<mlir::omp::DeclareTargetCaptureClause>(symClause),
        clauseOps.deviceType);
  }
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPRequiresConstruct &requiresConstruct) {
  // Requires directives are gathered and processed in semantics and
  // then combined in the lowering bridge before triggering codegen
  // just once. Hence, there is no need to lower each individual
  // occurrence here.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPThreadprivate &threadprivate) {
  // The directive is lowered when instantiating the variable to
  // support the case of threadprivate variable declared in module.
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {
  std::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      ompDeclConstruct.u);
}

//===----------------------------------------------------------------------===//
// OpenMPStandaloneConstruct visitors
//===----------------------------------------------------------------------===//

static void genOMP(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPSimpleStandaloneConstruct &simpleStandaloneConstruct) {
  const auto &directive = std::get<parser::OmpSimpleStandaloneDirective>(
      simpleStandaloneConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(simpleStandaloneConstruct.t), semaCtx);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, directive.source, directive.v, clauses)};
  if (directive.v == llvm::omp::Directive::OMPD_ordered) {
    // Standalone "ordered" directive.
    genOrderedOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
  } else {
    // Dispatch handles the "block-associated" variant of "ordered".
    genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                   queue.begin());
  }
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPFlushConstruct &flushConstruct) {
  const auto &verbatim = std::get<parser::Verbatim>(flushConstruct.t);
  const auto &objectList =
      std::get<std::optional<parser::OmpObjectList>>(flushConstruct.t);
  const auto &clauseList =
      std::get<std::optional<std::list<parser::OmpMemoryOrderClause>>>(
          flushConstruct.t);
  ObjectList objects =
      objectList ? makeObjects(*objectList, semaCtx) : ObjectList{};
  List<Clause> clauses =
      clauseList ? makeList(*clauseList,
                            [&](auto &&s) { return makeClause(s.v, semaCtx); })
                 : List<Clause>{};
  mlir::Location currentLocation = converter.genLocation(verbatim.source);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, verbatim.source,
      llvm::omp::Directive::OMPD_flush, clauses)};
  genFlushOp(converter, symTable, semaCtx, eval, currentLocation, objects,
             queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancelConstruct &cancelConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCancellationPointConstruct
                       &cancellationPointConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
}

static void
genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
       semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
       const parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      standaloneConstruct.u);
}

//===----------------------------------------------------------------------===//
// OpenMPConstruct visitors
//===----------------------------------------------------------------------===//

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAllocatorsConstruct &allocsConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(
      common::visitors{
          [&](const parser::OmpAtomicRead &atomicRead) {
            mlir::Location loc = converter.genLocation(atomicRead.source);
            lower::genOmpAccAtomicRead<parser::OmpAtomicRead,
                                       parser::OmpAtomicClauseList>(
                converter, atomicRead, loc);
          },
          [&](const parser::OmpAtomicWrite &atomicWrite) {
            mlir::Location loc = converter.genLocation(atomicWrite.source);
            lower::genOmpAccAtomicWrite<parser::OmpAtomicWrite,
                                        parser::OmpAtomicClauseList>(
                converter, atomicWrite, loc);
          },
          [&](const parser::OmpAtomic &atomicConstruct) {
            mlir::Location loc = converter.genLocation(atomicConstruct.source);
            lower::genOmpAtomic<parser::OmpAtomic, parser::OmpAtomicClauseList>(
                converter, atomicConstruct, loc);
          },
          [&](const parser::OmpAtomicUpdate &atomicUpdate) {
            mlir::Location loc = converter.genLocation(atomicUpdate.source);
            lower::genOmpAccAtomicUpdate<parser::OmpAtomicUpdate,
                                         parser::OmpAtomicClauseList>(
                converter, atomicUpdate, loc);
          },
          [&](const parser::OmpAtomicCapture &atomicCapture) {
            mlir::Location loc = converter.genLocation(atomicCapture.source);
            lower::genOmpAccAtomicCapture<parser::OmpAtomicCapture,
                                          parser::OmpAtomicClauseList>(
                converter, atomicCapture, loc);
          },
      },
      atomicConstruct.u);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &endBlockDirective =
      std::get<parser::OmpEndBlockDirective>(blockConstruct.t);
  mlir::Location currentLocation =
      converter.genLocation(beginBlockDirective.source);
  const auto origDirective =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).v;
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginBlockDirective.t), semaCtx);
  clauses.append(makeClauses(
      std::get<parser::OmpClauseList>(endBlockDirective.t), semaCtx));

  assert(llvm::omp::blockConstructSet.test(origDirective) &&
         "Expected block construct");
  (void)origDirective;

  for (const Clause &clause : clauses) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::holds_alternative<clause::Allocate>(clause.u) &&
        !std::holds_alternative<clause::Copyin>(clause.u) &&
        !std::holds_alternative<clause::Copyprivate>(clause.u) &&
        !std::holds_alternative<clause::Default>(clause.u) &&
        !std::holds_alternative<clause::Depend>(clause.u) &&
        !std::holds_alternative<clause::Final>(clause.u) &&
        !std::holds_alternative<clause::Firstprivate>(clause.u) &&
        !std::holds_alternative<clause::HasDeviceAddr>(clause.u) &&
        !std::holds_alternative<clause::If>(clause.u) &&
        !std::holds_alternative<clause::IsDevicePtr>(clause.u) &&
        !std::holds_alternative<clause::Map>(clause.u) &&
        !std::holds_alternative<clause::Nowait>(clause.u) &&
        !std::holds_alternative<clause::NumTeams>(clause.u) &&
        !std::holds_alternative<clause::NumThreads>(clause.u) &&
        !std::holds_alternative<clause::Priority>(clause.u) &&
        !std::holds_alternative<clause::Private>(clause.u) &&
        !std::holds_alternative<clause::ProcBind>(clause.u) &&
        !std::holds_alternative<clause::Reduction>(clause.u) &&
        !std::holds_alternative<clause::Shared>(clause.u) &&
        !std::holds_alternative<clause::Simd>(clause.u) &&
        !std::holds_alternative<clause::ThreadLimit>(clause.u) &&
        !std::holds_alternative<clause::Threads>(clause.u) &&
        !std::holds_alternative<clause::UseDeviceAddr>(clause.u) &&
        !std::holds_alternative<clause::UseDevicePtr>(clause.u)) {
      TODO(clauseLocation, "OpenMP Block construct clause");
    }
  }

  llvm::omp::Directive directive =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpBlockDirective>(beginBlockDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPCriticalConstruct &criticalConstruct) {
  const auto &cd = std::get<parser::OmpCriticalDirective>(criticalConstruct.t);
  List<Clause> clauses =
      makeClauses(std::get<parser::OmpClauseList>(cd.t), semaCtx);

  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval, cd.source,
      llvm::omp::Directive::OMPD_critical, clauses)};

  const auto &name = std::get<std::optional<parser::Name>>(cd.t);
  mlir::Location currentLocation = converter.getCurrentLocation();
  genCriticalOp(converter, symTable, semaCtx, eval, currentLocation, queue,
                queue.begin(), name);
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPExecutableAllocate &execAllocConstruct) {
  TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPLoopConstruct &loopConstruct) {
  const auto &beginLoopDirective =
      std::get<parser::OmpBeginLoopDirective>(loopConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginLoopDirective.t), semaCtx);
  if (auto &endLoopDirective =
          std::get<std::optional<parser::OmpEndLoopDirective>>(
              loopConstruct.t)) {
    clauses.append(makeClauses(
        std::get<parser::OmpClauseList>(endLoopDirective->t), semaCtx));
  }

  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);

  llvm::omp::Directive directive =
      std::get<parser::OmpLoopDirective>(beginLoopDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpLoopDirective>(beginLoopDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSectionConstruct &sectionConstruct) {
  mlir::Location loc = converter.getCurrentLocation();
  ConstructQueue queue{buildConstructQueue(
      converter.getFirOpBuilder().getModule(), semaCtx, eval,
      sectionConstruct.source, llvm::omp::Directive::OMPD_section, {})};
  genOMPDispatch(converter, symTable, semaCtx, eval, loc, queue, queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPSectionsConstruct &sectionsConstruct) {
  const auto &beginSectionsDirective =
      std::get<parser::OmpBeginSectionsDirective>(sectionsConstruct.t);
  List<Clause> clauses = makeClauses(
      std::get<parser::OmpClauseList>(beginSectionsDirective.t), semaCtx);
  const auto &endSectionsDirective =
      std::get<parser::OmpEndSectionsDirective>(sectionsConstruct.t);
  clauses.append(makeClauses(
      std::get<parser::OmpClauseList>(endSectionsDirective.t), semaCtx));
  mlir::Location currentLocation = converter.getCurrentLocation();

  llvm::omp::Directive directive =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).v;
  const parser::CharBlock &source =
      std::get<parser::OmpSectionsDirective>(beginSectionsDirective.t).source;
  ConstructQueue queue{
      buildConstructQueue(converter.getFirOpBuilder().getModule(), semaCtx,
                          eval, source, directive, clauses)};
  genOMPDispatch(converter, symTable, semaCtx, eval, currentLocation, queue,
                 queue.begin());
}

static void genOMP(lower::AbstractConverter &converter, lower::SymMap &symTable,
                   semantics::SemanticsContext &semaCtx,
                   lower::pft::Evaluation &eval,
                   const parser::OpenMPConstruct &ompConstruct) {
  std::visit(
      [&](auto &&s) { return genOMP(converter, symTable, semaCtx, eval, s); },
      ompConstruct.u);
}

//===----------------------------------------------------------------------===//
// Public functions
//===----------------------------------------------------------------------===//

mlir::Operation *Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                                     mlir::Operation *op,
                                                     mlir::Location loc) {
  if (mlir::isa<mlir::omp::AtomicUpdateOp, mlir::omp::DeclareReductionOp,
                mlir::omp::LoopNestOp>(op))
    return builder.create<mlir::omp::YieldOp>(loc);
  return builder.create<mlir::omp::TerminatorOp>(loc);
}

void Fortran::lower::genOpenMPConstruct(lower::AbstractConverter &converter,
                                        lower::SymMap &symTable,
                                        semantics::SemanticsContext &semaCtx,
                                        lower::pft::Evaluation &eval,
                                        const parser::OpenMPConstruct &omp) {
  symTable.pushScope();
  genOMP(converter, symTable, semaCtx, eval, omp);
  symTable.popScope();
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    lower::AbstractConverter &converter, lower::SymMap &symTable,
    semantics::SemanticsContext &semaCtx, lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &omp) {
  genOMP(converter, symTable, semaCtx, eval, omp);
  genNestedEvaluations(converter, eval);
}

void Fortran::lower::genOpenMPSymbolProperties(
    lower::AbstractConverter &converter, const lower::pft::Variable &var) {
  assert(var.hasSymbol() && "Expecting Symbol");
  const semantics::Symbol &sym = var.getSymbol();

  if (sym.test(semantics::Symbol::Flag::OmpThreadprivate))
    lower::genThreadprivateOp(converter, var);

  if (sym.test(semantics::Symbol::Flag::OmpDeclareTarget))
    lower::genDeclareTargetIntGlobal(converter, var);
}

int64_t
Fortran::lower::getCollapseValue(const parser::OmpClauseList &clauseList) {
  for (const parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = semantics::GetExpr(collapseClause->v);
      return evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

void Fortran::lower::genThreadprivateOp(lower::AbstractConverter &converter,
                                        const lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const semantics::Symbol *common =
          semantics::FindCommonBlockContaining(sym.GetUltimate())) {
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
    lower::AbstractConverter &converter, const lower::pft::Variable &var) {
  if (!var.isGlobal()) {
    // A non-global variable which can be in a declare target directive must
    // be a variable in the main program, and it has the implicit SAVE
    // attribute. We create a GlobalOp for it to simplify the translation to
    // LLVM IR.
    globalInitialization(converter, converter.getFirOpBuilder(),
                         var.getSymbol(), var, converter.getCurrentLocation());
  }
}

bool Fortran::lower::isOpenMPTargetConstruct(
    const parser::OpenMPConstruct &omp) {
  llvm::omp::Directive dir = llvm::omp::Directive::OMPD_unknown;
  if (const auto *block = std::get_if<parser::OpenMPBlockConstruct>(&omp.u)) {
    const auto &begin = std::get<parser::OmpBeginBlockDirective>(block->t);
    dir = std::get<parser::OmpBlockDirective>(begin.t).v;
  } else if (const auto *loop =
                 std::get_if<parser::OpenMPLoopConstruct>(&omp.u)) {
    const auto &begin = std::get<parser::OmpBeginLoopDirective>(loop->t);
    dir = std::get<parser::OmpLoopDirective>(begin.t).v;
  }
  return llvm::omp::allTargetSet.test(dir);
}

void Fortran::lower::gatherOpenMPDeferredDeclareTargets(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &ompDecl,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  std::visit(common::visitors{
                 [&](const parser::OpenMPDeclareTargetConstruct &ompReq) {
                   collectDeferredDeclareTargets(converter, semaCtx, eval,
                                                 ompReq, deferredDeclareTarget);
                 },
                 [&](const auto &) {},
             },
             ompDecl.u);
}

bool Fortran::lower::isOpenMPDeviceDeclareTarget(
    lower::AbstractConverter &converter, semantics::SemanticsContext &semaCtx,
    lower::pft::Evaluation &eval,
    const parser::OpenMPDeclarativeConstruct &ompDecl) {
  return std::visit(
      common::visitors{
          [&](const parser::OpenMPDeclareTargetConstruct &ompReq) {
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

void Fortran::lower::genOpenMPRequires(mlir::Operation *mod,
                                       const semantics::Symbol *symbol) {
  using MlirRequires = mlir::omp::ClauseRequires;
  using SemaRequires = semantics::WithOmpDeclarative::RequiresFlag;

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    semantics::WithOmpDeclarative::RequiresFlags semaFlags;
    if (symbol) {
      common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<semantics::WithOmpDeclarative,
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
