//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and optimisations of omp.workdistribute.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute"

using namespace mlir;

namespace {

static bool isRuntimeCall(Operation *op) {
  if (auto callOp = dyn_cast<fir::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    auto *func = op->getParentOfType<ModuleOp>().lookupSymbol(*callee);
    if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
      return true;
  }
  return false;
}

/// This is the single source of truth about whether we should parallelize an
/// operation nested in an omp.execute region.
static bool shouldParallelize(Operation *op) {
  if (llvm::any_of(op->getResults(),
                   [](OpResult v) -> bool { return !v.use_empty(); }))
    return false;
  // We will parallelize unordered loops - these come from array syntax
  if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
    auto unordered = loop.getUnordered();
    if (!unordered)
      return false;
    return *unordered;
  }
  if (isRuntimeCall(op)) {
    return true;
  }
  // We cannot parallise anything else
  return false;
}

template <typename T>
static T getPerfectlyNested(Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  auto &region = op->getRegion(0);
  if (region.getBlocks().size() != 1)
    return nullptr;
  auto *block = &region.front();
  auto *firstOp = &block->front();
  if (auto nested = dyn_cast<T>(firstOp))
    if (firstOp->getNextNode() == block->getTerminator())
      return nested;
  return nullptr;
}

/// If B() and D() are parallelizable,
///
/// omp.teams {
///   omp.workdistribute {
///     A()
///     B()
///     C()
///     D()
///     E()
///   }
/// }
///
/// becomes
///
/// A()
/// omp.teams {
///   omp.workdistribute {
///     B()
///   }
/// }
/// C()
/// omp.teams {
///   omp.workdistribute {
///     D()
///   }
/// }
/// E()

static bool FissionWorkdistribute(omp::WorkdistributeOp workdistribute) {
  OpBuilder rewriter(workdistribute);
  auto loc = workdistribute->getLoc();
  auto teams = dyn_cast<omp::TeamsOp>(workdistribute->getParentOp());
  if (!teams) {
    emitError(loc, "workdistribute not nested in teams\n");
    return false;
  }
  if (workdistribute.getRegion().getBlocks().size() != 1) {
    emitError(loc, "workdistribute with multiple blocks\n");
    return false;
  }
  if (teams.getRegion().getBlocks().size() != 1) {
    emitError(loc, "teams with multiple blocks\n");
    return false;
  }

  auto *teamsBlock = &teams.getRegion().front();
  bool changed = false;
  // Move the ops inside teams and before workdistribute outside.
  IRMapping irMapping;
  llvm::SmallVector<Operation *> teamsHoisted;
  for (auto &op : teams.getOps()) {
    if (&op == workdistribute) {
      break;
    }
    if (shouldParallelize(&op)) {
      emitError(loc, "teams has parallelize ops before first workdistribute\n");
      return false;
    } else {
      rewriter.setInsertionPoint(teams);
      rewriter.clone(op, irMapping);
      teamsHoisted.push_back(&op);
      changed = true;
    }
  }
  for (auto *op : llvm::reverse(teamsHoisted)) {
    op->replaceAllUsesWith(irMapping.lookup(op));
    op->erase();
  }

  // While we have unhandled operations in the original workdistribute
  auto *workdistributeBlock = &workdistribute.getRegion().front();
  auto *terminator = workdistributeBlock->getTerminator();
  while (&workdistributeBlock->front() != terminator) {
    rewriter.setInsertionPoint(teams);
    IRMapping mapping;
    llvm::SmallVector<Operation *> hoisted;
    Operation *parallelize = nullptr;
    for (auto &op : workdistribute.getOps()) {
      if (&op == terminator) {
        break;
      }
      if (shouldParallelize(&op)) {
        parallelize = &op;
        break;
      } else {
        rewriter.clone(op, mapping);
        hoisted.push_back(&op);
        changed = true;
      }
    }

    for (auto *op : llvm::reverse(hoisted)) {
      op->replaceAllUsesWith(mapping.lookup(op));
      op->erase();
    }

    if (parallelize && hoisted.empty() &&
        parallelize->getNextNode() == terminator)
      break;
    if (parallelize) {
      auto newTeams = rewriter.cloneWithoutRegions(teams);
      auto *newTeamsBlock = rewriter.createBlock(
          &newTeams.getRegion(), newTeams.getRegion().begin(), {}, {});
      for (auto arg : teamsBlock->getArguments())
        newTeamsBlock->addArgument(arg.getType(), arg.getLoc());
      auto newWorkdistribute = rewriter.create<omp::WorkdistributeOp>(loc);
      rewriter.create<omp::TerminatorOp>(loc);
      rewriter.createBlock(&newWorkdistribute.getRegion(),
                           newWorkdistribute.getRegion().begin(), {}, {});
      auto *cloned = rewriter.clone(*parallelize);
      parallelize->replaceAllUsesWith(cloned);
      parallelize->erase();
      rewriter.create<omp::TerminatorOp>(loc);
      changed = true;
    }
  }
  return changed;
}

/// If fir.do_loop is present inside teams workdistribute
///
/// omp.teams {
///   omp.workdistribute {
///     fir.do_loop unoredered {
///       ...
///     }
///   }
/// }
///
/// Then, its lowered to
///
/// omp.teams {
///   omp.parallel {
///     omp.distribute {
///     omp.wsloop {
///       omp.loop_nest
///         ...
///       }
///     }
///   }
/// }

static void genParallelOp(Location loc, OpBuilder &rewriter, bool composite) {
  auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
  parallelOp.setComposite(composite);
  rewriter.createBlock(&parallelOp.getRegion());
  rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));
  return;
}

static void genDistributeOp(Location loc, OpBuilder &rewriter, bool composite) {
  mlir::omp::DistributeOperands distributeClauseOps;
  auto distributeOp =
      rewriter.create<mlir::omp::DistributeOp>(loc, distributeClauseOps);
  distributeOp.setComposite(composite);
  auto distributeBlock = rewriter.createBlock(&distributeOp.getRegion());
  rewriter.setInsertionPointToStart(distributeBlock);
  return;
}

static void
genLoopNestClauseOps(OpBuilder &rewriter, fir::DoLoopOp loop,
                     mlir::omp::LoopNestOperands &loopNestClauseOps) {
  assert(loopNestClauseOps.loopLowerBounds.empty() &&
         "Loop nest bounds were already emitted!");
  loopNestClauseOps.loopLowerBounds.push_back(loop.getLowerBound());
  loopNestClauseOps.loopUpperBounds.push_back(loop.getUpperBound());
  loopNestClauseOps.loopSteps.push_back(loop.getStep());
  loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
}

static void genWsLoopOp(mlir::OpBuilder &rewriter, fir::DoLoopOp doLoop,
                        const mlir::omp::LoopNestOperands &clauseOps,
                        bool composite) {

  auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(doLoop.getLoc());
  wsloopOp.setComposite(composite);
  rewriter.createBlock(&wsloopOp.getRegion());

  auto loopNestOp =
      rewriter.create<mlir::omp::LoopNestOp>(doLoop.getLoc(), clauseOps);

  // Clone the loop's body inside the loop nest construct using the
  // mapped values.
  rewriter.cloneRegionBefore(doLoop.getRegion(), loopNestOp.getRegion(),
                             loopNestOp.getRegion().begin());
  Block *clonedBlock = &loopNestOp.getRegion().back();
  mlir::Operation *terminatorOp = clonedBlock->getTerminator();

  // Erase fir.result op of do loop and create yield op.
  if (auto resultOp = dyn_cast<fir::ResultOp>(terminatorOp)) {
    rewriter.setInsertionPoint(terminatorOp);
    rewriter.create<mlir::omp::YieldOp>(doLoop->getLoc());
    // rewriter.erase(terminatorOp);
    terminatorOp->erase();
  }
  return;
}

static bool WorkdistributeDoLower(omp::WorkdistributeOp workdistribute) {
  OpBuilder rewriter(workdistribute);
  auto doLoop = getPerfectlyNested<fir::DoLoopOp>(workdistribute);
  auto wdLoc = workdistribute->getLoc();
  if (doLoop && shouldParallelize(doLoop)) {
    assert(doLoop.getReduceOperands().empty());
    genParallelOp(wdLoc, rewriter, true);
    genDistributeOp(wdLoc, rewriter, true);
    mlir::omp::LoopNestOperands loopNestClauseOps;
    genLoopNestClauseOps(rewriter, doLoop, loopNestClauseOps);
    genWsLoopOp(rewriter, doLoop, loopNestClauseOps, true);
    workdistribute.erase();
    return true;
  }
  return false;
}

/// If A() and B () are present inside teams workdistribute
///
/// omp.teams {
///   omp.workdistribute {
///     A()
///     B()
///   }
/// }
///
/// Then, its lowered to
///
/// A()
/// B()
///

static bool TeamsWorkdistributeToSingleOp(omp::TeamsOp teamsOp) {
  auto workdistributeOp = getPerfectlyNested<omp::WorkdistributeOp>(teamsOp);
  if (!workdistributeOp)
    return false;
  // Get the block containing teamsOp (the parent block).
  Block *parentBlock = teamsOp->getBlock();
  Block &workdistributeBlock = *workdistributeOp.getRegion().begin();
  auto insertPoint = Block::iterator(teamsOp);
  // Get the range of operations to move (excluding the terminator).
  auto workdistributeBegin = workdistributeBlock.begin();
  auto workdistributeEnd = workdistributeBlock.getTerminator()->getIterator();
  // Move the operations from workdistribute block to before teamsOp.
  parentBlock->getOperations().splice(insertPoint,
                                      workdistributeBlock.getOperations(),
                                      workdistributeBegin, workdistributeEnd);
  // Erase the now-empty workdistributeOp.
  workdistributeOp.erase();
  Block &teamsBlock = *teamsOp.getRegion().begin();
  // Check if only the terminator remains and erase teams op.
  if (teamsBlock.getOperations().size() == 1 &&
      teamsBlock.getTerminator() != nullptr) {
    teamsOp.erase();
  }
  return true;
}

struct SplitTargetResult {
  omp::TargetOp targetOp;
  omp::TargetDataOp dataOp;
};

/// If multiple workdistribute are nested in a target regions, we will need to
/// split the target region, but we want to preserve the data semantics of the
/// original data region and avoid unnecessary data movement at each of the
/// subkernels - we split the target region into a target_data{target}
/// nest where only the outer one moves the data
std::optional<SplitTargetResult> splitTargetData(omp::TargetOp targetOp,
                                                 RewriterBase &rewriter) {
  auto loc = targetOp->getLoc();
  if (targetOp.getMapVars().empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " target region has no data maps\n");
    return std::nullopt;
  }

  SmallVector<omp::MapInfoOp> mapInfos;
  for (auto opr : targetOp.getMapVars()) {
    auto mapInfo = cast<omp::MapInfoOp>(opr.getDefiningOp());
    mapInfos.push_back(mapInfo);
  }

  rewriter.setInsertionPoint(targetOp);
  SmallVector<Value> innerMapInfos;
  SmallVector<Value> outerMapInfos;

  for (auto mapInfo : mapInfos) {
    auto originalMapType =
        (llvm::omp::OpenMPOffloadMappingFlags)(mapInfo.getMapType());
    auto originalCaptureType = mapInfo.getMapCaptureType();
    llvm::omp::OpenMPOffloadMappingFlags newMapType;
    mlir::omp::VariableCaptureKind newCaptureType;

    if (originalCaptureType == mlir::omp::VariableCaptureKind::ByCopy) {
      newMapType = originalMapType;
      newCaptureType = originalCaptureType;
    } else if (originalCaptureType == mlir::omp::VariableCaptureKind::ByRef) {
      newMapType = llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
      newCaptureType = originalCaptureType;
      outerMapInfos.push_back(mapInfo);
    } else {
      llvm_unreachable("Unhandled case");
    }
    auto innerMapInfo = cast<omp::MapInfoOp>(rewriter.clone(*mapInfo));
    innerMapInfo.setMapTypeAttr(rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, false),
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            newMapType)));
    innerMapInfo.setMapCaptureType(newCaptureType);
    innerMapInfos.push_back(innerMapInfo.getResult());
  }

  rewriter.setInsertionPoint(targetOp);
  auto device = targetOp.getDevice();
  auto ifExpr = targetOp.getIfExpr();
  auto deviceAddrVars = targetOp.getHasDeviceAddrVars();
  auto devicePtrVars = targetOp.getIsDevicePtrVars();
  auto targetDataOp = rewriter.create<omp::TargetDataOp>(
      loc, device, ifExpr, outerMapInfos, deviceAddrVars, devicePtrVars);
  auto taregtDataBlock = rewriter.createBlock(&targetDataOp.getRegion());
  rewriter.create<mlir::omp::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(taregtDataBlock);

  auto newTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
      targetOp.getHostEvalVars(), targetOp.getIfExpr(),
      targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
      targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
      innerMapInfos, targetOp.getNowaitAttr(), targetOp.getPrivateVars(),
      targetOp.getPrivateSymsAttr(), targetOp.getPrivateNeedsBarrierAttr(),
      targetOp.getThreadLimit(), targetOp.getPrivateMapsAttr());
  rewriter.inlineRegionBefore(targetOp.getRegion(), newTargetOp.getRegion(),
                              newTargetOp.getRegion().begin());

  rewriter.replaceOp(targetOp, targetDataOp);
  return SplitTargetResult{cast<omp::TargetOp>(newTargetOp), targetDataOp};
}

static std::optional<std::tuple<Operation *, bool, bool>>
getNestedOpToIsolate(omp::TargetOp targetOp) {
  if (targetOp.getRegion().empty())
    return std::nullopt;
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto &op : *targetBlock) {
    bool first = &op == &*targetBlock->begin();
    bool last = op.getNextNode() == targetBlock->getTerminator();
    if (first && last)
      return std::nullopt;

    if (isa<omp::TeamsOp, omp::ParallelOp>(&op))
      return {{&op, first, last}};
  }
  return std::nullopt;
}

struct TempOmpVar {
  omp::MapInfoOp from, to;
};

static bool isPtr(Type ty) {
  return isa<fir::ReferenceType>(ty) || isa<LLVM::LLVMPointerType>(ty);
}

static Type getPtrTypeForOmp(Type ty) {
  if (isPtr(ty))
    return LLVM::LLVMPointerType::get(ty.getContext());
  else
    return fir::LLVMPointerType::get(ty);
}

static TempOmpVar allocateTempOmpVar(Location loc, Type ty,
                                     RewriterBase &rewriter) {
  MLIRContext &ctx = *ty.getContext();
  Value alloc;
  Type allocType;
  auto llvmPtrTy = LLVM::LLVMPointerType::get(&ctx);
  if (isPtr(ty)) {
    Type intTy = rewriter.getI32Type();
    auto one = rewriter.create<LLVM::ConstantOp>(loc, intTy, 1);
    allocType = llvmPtrTy;
    alloc = rewriter.create<LLVM::AllocaOp>(loc, llvmPtrTy, allocType, one);
    allocType = intTy;
  } else {
    allocType = ty;
    alloc = rewriter.create<fir::AllocaOp>(loc, allocType);
  }
  auto getMapInfo = [&](uint64_t mappingFlags, const char *name) {
    return rewriter.create<omp::MapInfoOp>(
        loc, alloc.getType(), alloc, TypeAttr::get(allocType),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/false),
                                mappingFlags),
        rewriter.getAttr<omp::VariableCaptureKindAttr>(
            omp::VariableCaptureKind::ByRef),
        /*varPtrPtr=*/Value{},
        /*members=*/SmallVector<Value>{},
        /*member_index=*/mlir::ArrayAttr{},
        /*bounds=*/ValueRange(),
        /*mapperId=*/mlir::FlatSymbolRefAttr(),
        /*name=*/rewriter.getStringAttr(name), rewriter.getBoolAttr(false));
  };
  uint64_t mapFrom =
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM);
  uint64_t mapTo =
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
  auto mapInfoFrom = getMapInfo(mapFrom, "__flang_workdistribute_from");
  auto mapInfoTo = getMapInfo(mapTo, "__flang_workdistribute_to");
  return TempOmpVar{mapInfoFrom, mapInfoTo};
};

static bool usedOutsideSplit(Value v, Operation *split) {
  if (!split)
    return false;
  auto targetOp = cast<omp::TargetOp>(split->getParentOp());
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto *user : v.getUsers()) {
    while (user->getBlock() != targetBlock) {
      user = user->getParentOp();
    }
    if (!user->isBeforeInBlock(split))
      return true;
  }
  return false;
};

static bool isRecomputableAfterFission(Operation *op, Operation *splitBefore) {
  if (isa<fir::DeclareOp>(op))
    return true;

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    return false;
  }
  interface.getEffects(effects);
  if (effects.empty())
    return true;
  return false;
}

struct SplitResult {
  omp::TargetOp preTargetOp;
  omp::TargetOp isolatedTargetOp;
  omp::TargetOp postTargetOp;
};

static void collectNonRecomputableDeps(Value &v, omp::TargetOp targetOp,
                                       SetVector<Operation *> &nonRecomputable,
                                       SetVector<Operation *> &toCache,
                                       SetVector<Operation *> &toRecompute) {
  Operation *op = v.getDefiningOp();
  if (!op) {
    assert(cast<BlockArgument>(v).getOwner()->getParentOp() == targetOp);
    return;
  }
  if (nonRecomputable.contains(op)) {
    toCache.insert(op);
    return;
  }
  toRecompute.insert(op);
  for (auto opr : op->getOperands())
    collectNonRecomputableDeps(opr, targetOp, nonRecomputable, toCache,
                               toRecompute);
}

static void reloadCacheAndRecompute(Location loc, RewriterBase &rewriter,
                                    MLIRContext &ctx, IRMapping &mapping,
                                    Operation *splitBefore, Block *targetBlock,
                                    Block *newTargetBlock,
                                    SmallVector<Value> &allocs,
                                    SetVector<Operation *> &toRecompute) {
  for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
    auto originalArg = targetBlock->getArgument(i);
    auto newArg = newTargetBlock->addArgument(originalArg.getType(),
                                              originalArg.getLoc());
    mapping.map(originalArg, newArg);
  }
  auto llvmPtrTy = LLVM::LLVMPointerType::get(&ctx);
  for (auto original : allocs) {
    Value newArg = newTargetBlock->addArgument(
        getPtrTypeForOmp(original.getType()), original.getLoc());
    Value restored;
    if (isPtr(original.getType())) {
      restored = rewriter.create<LLVM::LoadOp>(loc, llvmPtrTy, newArg);
      if (!isa<LLVM::LLVMPointerType>(original.getType()))
        restored =
            rewriter.create<fir::ConvertOp>(loc, original.getType(), restored);
    } else {
      restored = rewriter.create<fir::LoadOp>(loc, newArg);
    }
    mapping.map(original, restored);
  }
  for (auto it = targetBlock->begin(); it != splitBefore->getIterator(); it++) {
    if (toRecompute.contains(&*it))
      rewriter.clone(*it, mapping);
  }
}

static SplitResult isolateOp(Operation *splitBeforeOp, bool splitAfter,
                             RewriterBase &rewriter) {
  auto targetOp = cast<omp::TargetOp>(splitBeforeOp->getParentOp());
  MLIRContext &ctx = *targetOp.getContext();
  assert(targetOp);
  auto loc = targetOp.getLoc();
  auto *targetBlock = &targetOp.getRegion().front();
  rewriter.setInsertionPoint(targetOp);

  auto preMapOperands = SmallVector<Value>(targetOp.getMapVars());
  auto postMapOperands = SmallVector<Value>(targetOp.getMapVars());

  SmallVector<Value> requiredVals;
  SetVector<Operation *> toCache;
  SetVector<Operation *> toRecompute;
  SetVector<Operation *> nonRecomputable;
  SmallVector<Value> allocs;

  for (auto it = targetBlock->begin(); it != splitBeforeOp->getIterator();
       it++) {
    for (auto res : it->getResults()) {
      if (usedOutsideSplit(res, splitBeforeOp))
        requiredVals.push_back(res);
    }
    if (!isRecomputableAfterFission(&*it, splitBeforeOp))
      nonRecomputable.insert(&*it);
  }

  for (auto requiredVal : requiredVals)
    collectNonRecomputableDeps(requiredVal, targetOp, nonRecomputable, toCache,
                               toRecompute);

  for (Operation *op : toCache) {
    for (auto res : op->getResults()) {
      auto alloc =
          allocateTempOmpVar(targetOp.getLoc(), res.getType(), rewriter);
      allocs.push_back(res);
      preMapOperands.push_back(alloc.from);
      postMapOperands.push_back(alloc.to);
    }
  }

  rewriter.setInsertionPoint(targetOp);

  auto preTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
      targetOp.getHostEvalVars(), targetOp.getIfExpr(),
      targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
      targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
      preMapOperands, targetOp.getNowaitAttr(), targetOp.getPrivateVars(),
      targetOp.getPrivateSymsAttr(), targetOp.getPrivateNeedsBarrierAttr(),
      targetOp.getThreadLimit(), targetOp.getPrivateMapsAttr());
  auto *preTargetBlock = rewriter.createBlock(
      &preTargetOp.getRegion(), preTargetOp.getRegion().begin(), {}, {});
  IRMapping preMapping;
  for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
    auto originalArg = targetBlock->getArgument(i);
    auto newArg = preTargetBlock->addArgument(originalArg.getType(),
                                              originalArg.getLoc());
    preMapping.map(originalArg, newArg);
  }
  for (auto it = targetBlock->begin(); it != splitBeforeOp->getIterator(); it++)
    rewriter.clone(*it, preMapping);

  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

  for (auto original : allocs) {
    Value toStore = preMapping.lookup(original);
    auto newArg = preTargetBlock->addArgument(
        getPtrTypeForOmp(original.getType()), original.getLoc());
    if (isPtr(original.getType())) {
      if (!isa<LLVM::LLVMPointerType>(toStore.getType()))
        toStore = rewriter.create<fir::ConvertOp>(loc, llvmPtrTy, toStore);
      rewriter.create<LLVM::StoreOp>(loc, toStore, newArg);
    } else {
      rewriter.create<fir::StoreOp>(loc, toStore, newArg);
    }
  }
  rewriter.create<omp::TerminatorOp>(loc);

  rewriter.setInsertionPoint(targetOp);

  auto isolatedTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
      targetOp.getHostEvalVars(), targetOp.getIfExpr(),
      targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
      targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
      postMapOperands, targetOp.getNowaitAttr(), targetOp.getPrivateVars(),
      targetOp.getPrivateSymsAttr(), targetOp.getPrivateNeedsBarrierAttr(),
      targetOp.getThreadLimit(), targetOp.getPrivateMapsAttr());

  auto *isolatedTargetBlock =
      rewriter.createBlock(&isolatedTargetOp.getRegion(),
                           isolatedTargetOp.getRegion().begin(), {}, {});

  IRMapping isolatedMapping;
  reloadCacheAndRecompute(loc, rewriter, ctx, isolatedMapping, splitBeforeOp,
                          targetBlock, isolatedTargetBlock, allocs,
                          toRecompute);
  rewriter.clone(*splitBeforeOp, isolatedMapping);
  rewriter.create<omp::TerminatorOp>(loc);

  omp::TargetOp postTargetOp = nullptr;

  if (splitAfter) {
    rewriter.setInsertionPoint(targetOp);
    postTargetOp = rewriter.create<omp::TargetOp>(
        targetOp.getLoc(), targetOp.getAllocateVars(),
        targetOp.getAllocatorVars(), targetOp.getBareAttr(),
        targetOp.getDependKindsAttr(), targetOp.getDependVars(),
        targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
        targetOp.getHostEvalVars(), targetOp.getIfExpr(),
        targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
        targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
        postMapOperands, targetOp.getNowaitAttr(), targetOp.getPrivateVars(),
        targetOp.getPrivateSymsAttr(), targetOp.getPrivateNeedsBarrierAttr(),
        targetOp.getThreadLimit(), targetOp.getPrivateMapsAttr());
    auto *postTargetBlock = rewriter.createBlock(
        &postTargetOp.getRegion(), postTargetOp.getRegion().begin(), {}, {});
    IRMapping postMapping;
    reloadCacheAndRecompute(loc, rewriter, ctx, postMapping, splitBeforeOp,
                            targetBlock, postTargetBlock, allocs, toRecompute);

    assert(splitBeforeOp->getNumResults() == 0 ||
           llvm::all_of(splitBeforeOp->getResults(),
                        [](Value result) { return result.use_empty(); }));

    for (auto it = std::next(splitBeforeOp->getIterator());
         it != targetBlock->end(); it++)
      rewriter.clone(*it, postMapping);
  }

  rewriter.eraseOp(targetOp);
  return SplitResult{preTargetOp, isolatedTargetOp, postTargetOp};
}

static mlir::LLVM::ConstantOp
genI32Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

static Type getOmpDeviceType(MLIRContext *c) { return IntegerType::get(c, 32); }

static void moveToHost(omp::TargetOp targetOp, RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *targetBlock = &targetOp.getRegion().front();
  assert(targetBlock == &targetOp.getRegion().back());
  IRMapping mapping;
  for (auto map :
       zip_equal(targetOp.getMapVars(), targetBlock->getArguments())) {
    Value mapInfo = std::get<0>(map);
    BlockArgument arg = std::get<1>(map);
    Operation *op = mapInfo.getDefiningOp();
    assert(op);
    auto mapInfoOp = cast<omp::MapInfoOp>(op);
    mapping.map(arg, mapInfoOp.getVarPtr());
  }
  rewriter.setInsertionPoint(targetOp);
  SmallVector<Operation *> opsToReplace;
  Value device = targetOp.getDevice();
  if (!device) {
    device = genI32Constant(targetOp.getLoc(), rewriter, 0);
  }
  for (auto it = targetBlock->begin(), end = std::prev(targetBlock->end());
       it != end; ++it) {
    auto *op = &*it;
    if (isRuntimeCall(op)) {
      fir::CallOp runtimeCall = cast<fir::CallOp>(op);
      auto module = runtimeCall->getParentOfType<ModuleOp>();
      auto callee =
          cast<func::FuncOp>(module.lookupSymbol(runtimeCall.getCalleeAttr()));
      std::string newCalleeName = (callee.getName() + "_omp").str();
      mlir::OpBuilder moduleBuilder(module.getBodyRegion());
      func::FuncOp newCallee =
          cast_or_null<func::FuncOp>(module.lookupSymbol(newCalleeName));
      if (!newCallee) {
        SmallVector<Type> argTypes(callee.getFunctionType().getInputs());
        argTypes.push_back(getOmpDeviceType(rewriter.getContext()));
        newCallee = moduleBuilder.create<func::FuncOp>(
            callee->getLoc(), newCalleeName,
            FunctionType::get(rewriter.getContext(), argTypes,
                              callee.getFunctionType().getResults()));
        if (callee.getArgAttrs())
          newCallee.setArgAttrsAttr(*callee.getArgAttrs());
        if (callee.getResAttrs())
          newCallee.setResAttrsAttr(*callee.getResAttrs());
        newCallee.setSymVisibility(callee.getSymVisibility());
        newCallee->setDiscardableAttrs(callee->getDiscardableAttrDictionary());
      }
      SmallVector<Value> operands = runtimeCall.getOperands();
      operands.push_back(device);
      auto tmpCall = rewriter.create<fir::CallOp>(
          runtimeCall.getLoc(), runtimeCall.getResultTypes(),
          SymbolRefAttr::get(newCallee), operands, nullptr, nullptr, nullptr,
          runtimeCall.getFastmathAttr());
      Operation *newCall = rewriter.clone(*tmpCall, mapping);
      mapping.map(&*it, newCall);
      rewriter.eraseOp(tmpCall);
    } else {
      Operation *clonedOp = rewriter.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), clonedOp->getResult(i));
      }
      // fir.declare changes its type when hoisting it out of omp.target to
      // omp.target_data Introduce a load, if original declareOp input is not of
      // reference type, but cloned delcareOp input is reference type.
      if (fir::DeclareOp clonedDeclareOp = dyn_cast<fir::DeclareOp>(clonedOp)) {
        auto originalDeclareOp = cast<fir::DeclareOp>(op);
        Type originalInType = originalDeclareOp.getMemref().getType();
        Type clonedInType = clonedDeclareOp.getMemref().getType();

        fir::ReferenceType originalRefType =
            dyn_cast<fir::ReferenceType>(originalInType);
        fir::ReferenceType clonedRefType =
            dyn_cast<fir::ReferenceType>(clonedInType);
        if (!originalRefType && clonedRefType) {
          Type clonedEleTy = clonedRefType.getElementType();
          if (clonedEleTy == originalDeclareOp.getType()) {
            opsToReplace.push_back(clonedOp);
          }
        }
      }
      if (isa<fir::AllocMemOp>(clonedOp) || isa<fir::FreeMemOp>(clonedOp))
        opsToReplace.push_back(clonedOp);
    }
  }
  for (Operation *op : opsToReplace) {
    if (auto allocOp = dyn_cast<fir::AllocMemOp>(op)) {
      rewriter.setInsertionPoint(allocOp);
      auto ompAllocmemOp = rewriter.create<omp::TargetAllocMemOp>(
          allocOp.getLoc(), rewriter.getI64Type(), device,
          allocOp.getInTypeAttr(), allocOp.getUniqNameAttr(),
          allocOp.getBindcNameAttr(), allocOp.getTypeparams(),
          allocOp.getShape());
      auto firConvertOp = rewriter.create<fir::ConvertOp>(
          allocOp.getLoc(), allocOp.getResult().getType(),
          ompAllocmemOp.getResult());
      rewriter.replaceOp(allocOp, firConvertOp.getResult());
    } else if (auto freeOp = dyn_cast<fir::FreeMemOp>(op)) {
      rewriter.setInsertionPoint(freeOp);
      auto firConvertOp = rewriter.create<fir::ConvertOp>(
          freeOp.getLoc(), rewriter.getI64Type(), freeOp.getHeapref());
      rewriter.create<omp::TargetFreeMemOp>(freeOp.getLoc(), device,
                                            firConvertOp.getResult());
      rewriter.eraseOp(freeOp);
    } else if (fir::DeclareOp clonedDeclareOp = dyn_cast<fir::DeclareOp>(op)) {
      Type clonedInType = clonedDeclareOp.getMemref().getType();
      fir::ReferenceType clonedRefType =
          dyn_cast<fir::ReferenceType>(clonedInType);
      Type clonedEleTy = clonedRefType.getElementType();
      rewriter.setInsertionPoint(op);
      Value loadedValue = rewriter.create<fir::LoadOp>(
          clonedDeclareOp.getLoc(), clonedEleTy, clonedDeclareOp.getMemref());
      clonedDeclareOp.getResult().replaceAllUsesWith(loadedValue);
    }
  }
  rewriter.eraseOp(targetOp);
}

void fissionTarget(omp::TargetOp targetOp, RewriterBase &rewriter) {
  auto tuple = getNestedOpToIsolate(targetOp);
  if (!tuple) {
    LLVM_DEBUG(llvm::dbgs() << " No op to isolate\n");
    moveToHost(targetOp, rewriter);
    return;
  }

  Operation *toIsolate = std::get<0>(*tuple);
  bool splitBefore = !std::get<1>(*tuple);
  bool splitAfter = !std::get<2>(*tuple);

  if (splitBefore && splitAfter) {
    auto res = isolateOp(toIsolate, splitAfter, rewriter);
    moveToHost(res.preTargetOp, rewriter);
    fissionTarget(res.postTargetOp, rewriter);
    return;
  }
  if (splitBefore) {
    auto res = isolateOp(toIsolate, splitAfter, rewriter);
    moveToHost(res.preTargetOp, rewriter);
    return;
  }
  if (splitAfter) {
    auto res = isolateOp(toIsolate->getNextNode(), splitAfter, rewriter);
    fissionTarget(res.postTargetOp, rewriter);
    return;
  }
}

class LowerWorkdistributePass
    : public flangomp::impl::LowerWorkdistributeBase<LowerWorkdistributePass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    auto moduleOp = getOperation();
    bool changed = false;
    moduleOp->walk([&](mlir::omp::WorkdistributeOp workdistribute) {
      changed |= FissionWorkdistribute(workdistribute);
    });
    moduleOp->walk([&](mlir::omp::WorkdistributeOp workdistribute) {
      changed |= WorkdistributeDoLower(workdistribute);
    });
    moduleOp->walk([&](mlir::omp::TeamsOp teams) {
      changed |= TeamsWorkdistributeToSingleOp(teams);
    });

    if (changed) {
      SmallVector<omp::TargetOp> targetOps;
      moduleOp->walk(
          [&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
      IRRewriter rewriter(&context);
      for (auto targetOp : targetOps) {
        auto res = splitTargetData(targetOp, rewriter);
        if (res)
          fissionTarget(res->targetOp, rewriter);
      }
    }
  }
};
} // namespace
