//===- LowerWorkdistribute.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and optimisations of omp.workdistribute.
//
// Fortran array statements are lowered to fir as fir.do_loop unordered.
// lower-workdistribute pass works mainly on identifying fir.do_loop unordered
// that is nested in target{teams{workdistribute{fir.do_loop unordered}}} and
// lowers it to target{teams{parallel{wsloop{loop_nest}}}}.
// It hoists all the other ops outside target region.
// Relaces heap allocation on target with omp.target_allocmem and
// deallocation with omp.target_freemem from host. Also replaces
// runtime function "Assign" with equivalent omp function. ex. @_FortranAAssign
// on target, once hoisted outside target is replaced with @_FortranAAssign_omp.
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

// The isRuntimeCall function is a utility designed to determine
// if a given operation is a call to a Fortran-specific runtime function.
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

// This is the single source of truth about whether we should parallelize an
// operation nested in an omp.execute region.
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
  if (isRuntimeCall(op) &&
      (op->getName().getStringRef() == "_FortranAAssign")) {
    return true;
  }
  // We cannot parallise anything else.
  return false;
}

// The getPerfectlyNested function is a generic utility for finding
// a single, "perfectly nested" operation within a parent operation.
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

// FissionWorkdistribute method finds the parallelizable ops
// within teams {workdistribute} region and moves them to their
// own teams{workdistribute} region.
//
// If B() and D() are parallelizable,
//
// omp.teams {
//   omp.workdistribute {
//     A()
//     B()
//     C()
//     D()
//     E()
//   }
// }
//
// becomes
//
// A()
// omp.teams {
//   omp.workdistribute {
//     B()
//   }
// }
// C()
// omp.teams {
//   omp.workdistribute {
//     D()
//   }
// }
// E()

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

// WorkdistributeDoLower method finds the fir.do_loop unoredered
// nested in teams {workdistribute{fir.do_loop unoredered}} and
// lowers it to teams {parallel { distribute {wsloop {loop_nest}}}}.
//
// If fir.do_loop is present inside teams workdistribute
//
// omp.teams {
//   omp.workdistribute {
//     fir.do_loop unoredered {
//       ...
//     }
//   }
// }
//
// Then, its lowered to
//
// omp.teams {
//   omp.parallel {
//     omp.distribute {
//     omp.wsloop {
//       omp.loop_nest
//         ...
//       }
//     }
//   }
// }

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

// TeamsWorkdistributeToSingleOp method hoists all the ops inside
// teams {workdistribute{}} before teams op.
//
// If A() and B () are present inside teams workdistribute
//
// omp.teams {
//   omp.workdistribute {
//     A()
//     B()
//   }
// }
//
// Then, its lowered to
//
// A()
// B()
//

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

// If multiple workdistribute are nested in a target regions, we will need to
// split the target region, but we want to preserve the data semantics of the
// original data region and avoid unnecessary data movement at each of the
// subkernels - we split the target region into a target_data{target}
// nest where only the outer one moves the data
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

// getNestedOpToIsolate function is designed to identify a specific teams
// parallel op within the body of an omp::TargetOp that should be "isolated."
// This returns a tuple of op, if its first op in targetBlock, or if the op is
// last op in the tragte block.
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

static void createBlockArgsAndMap(Location loc, RewriterBase &rewriter,
                                  omp::TargetOp &targetOp, Block *targetBlock,
                                  Block *newTargetBlock,
                                  SmallVector<Value> &mapOperands,
                                  SmallVector<Value> &allocs,
                                  IRMapping &irMapping) {
  // Map `map_operands` to block arguments.
  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  for (unsigned i = 0; i < mapOperands.size(); ++i) {
    Value originalValue;
    BlockArgument newArg;
    // Map the new arguments from the original block.
    if (i < originalMapVarsSize) {
      originalValue = targetBlock->getArgument(i);
      newArg = newTargetBlock->addArgument(originalValue.getType(),
                                           originalValue.getLoc());
    }
    // Map the new arguments from the `allocs`.
    else {
      originalValue = allocs[i - originalMapVarsSize];
      newArg = newTargetBlock->addArgument(
          getPtrTypeForOmp(originalValue.getType()), originalValue.getLoc());
    }
    irMapping.map(originalValue, newArg);
  }
  // Map `private_vars` to block arguments.
  unsigned originalPrivateVarsSize = targetOp.getPrivateVars().size();
  for (unsigned i = 0; i < originalPrivateVarsSize; ++i) {
    auto originalArg = targetBlock->getArgument(originalMapVarsSize + i);
    auto newArg = newTargetBlock->addArgument(originalArg.getType(),
                                              originalArg.getLoc());
    irMapping.map(originalArg, newArg);
  }
  return;
}

static void reloadCacheAndRecompute(
    Location loc, RewriterBase &rewriter, Operation *splitBefore,
    omp::TargetOp &targetOp, Block *targetBlock, Block *newTargetBlock,
    SmallVector<Value> &mapOperands, SmallVector<Value> &allocs,
    SetVector<Operation *> &toRecompute, IRMapping &irMapping) {
  createBlockArgsAndMap(loc, rewriter, targetOp, targetBlock, newTargetBlock,
                        mapOperands, allocs, irMapping);
  // Handle the load operations for the allocs.
  rewriter.setInsertionPointToStart(newTargetBlock);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  // Create Stores for allocs.
  for (unsigned i = 0; i < allocs.size(); ++i) {
    Value original = allocs[i];
    // Get the new block argument for this specific allocated value.
    Value newArg = newTargetBlock->getArgument(originalMapVarsSize + i);

    Value restored;
    if (isPtr(original.getType())) {
      restored = rewriter.create<LLVM::LoadOp>(loc, llvmPtrTy, newArg);
      if (!isa<LLVM::LLVMPointerType>(original.getType()))
        restored =
            rewriter.create<fir::ConvertOp>(loc, original.getType(), restored);
    } else {
      restored = rewriter.create<fir::LoadOp>(loc, newArg);
    }
    irMapping.map(original, restored);
  }

  for (auto it = targetBlock->begin(); it != splitBefore->getIterator(); it++) {
    if (toRecompute.contains(&*it))
      rewriter.clone(*it, irMapping);
  }
}

// isolateOp method rewrites a omp.target_data { omp.target } in to
// omp.target_data {
//      // preTargetOp region contains ops before splitBeforeOp.
//      omp.target {}
//      // isolatedTargetOp region contains splitBeforeOp,
//      omp.target {}
//      // postTargetOp region contains ops after splitBeforeOp.
//      omp.target {}
// }
static SplitResult isolateOp(Operation *splitBeforeOp, bool splitAfter,
                             RewriterBase &rewriter) {
  auto targetOp = cast<omp::TargetOp>(splitBeforeOp->getParentOp());
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

  createBlockArgsAndMap(loc, rewriter, targetOp, targetBlock, preTargetBlock,
                        preMapOperands, allocs, preMapping);

  // Handle the store operations for the allocs.
  rewriter.setInsertionPointToStart(preTargetBlock);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

  // Clone the original operations.
  for (auto it = targetBlock->begin(); it != splitBeforeOp->getIterator();
       it++) {
    rewriter.clone(*it, preMapping);
  }

  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  // Create Stores for allocs.
  for (unsigned i = 0; i < allocs.size(); ++i) {
    Value originalResult = allocs[i];
    Value toStore = preMapping.lookup(originalResult);
    // Get the new block argument for this specific allocated value.
    Value newArg = preTargetBlock->getArgument(originalMapVarsSize + i);

    if (isPtr(originalResult.getType())) {
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
  reloadCacheAndRecompute(loc, rewriter, splitBeforeOp, targetOp, targetBlock,
                          isolatedTargetBlock, postMapOperands, allocs,
                          toRecompute, isolatedMapping);
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
    reloadCacheAndRecompute(loc, rewriter, splitBeforeOp, targetOp, targetBlock,
                            postTargetBlock, postMapOperands, allocs,
                            toRecompute, postMapping);

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

static mlir::LLVM::ConstantOp
genI64Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i64Ty = rewriter.getI64Type();
  mlir::IntegerAttr attr = rewriter.getI64IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, attr);
}

static Value genDescriptorGetBaseAddress(fir::FirOpBuilder &builder,
                                         Location loc, Value boxDesc) {
  Value box = boxDesc;
  if (auto refBox = dyn_cast<fir::ReferenceType>(boxDesc.getType())) {
    box = fir::LoadOp::create(builder, loc, boxDesc);
  }
  assert(isa<fir::BoxType>(box.getType()) &&
         "Unknown type passed to genDescriptorGetBaseAddress");
  auto i8Type = builder.getI8Type();
  auto unknownArrayType =
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, i8Type);
  auto i8BoxType = fir::BoxType::get(unknownArrayType);
  auto typedBox = fir::ConvertOp::create(builder, loc, i8BoxType, box);
  auto rawAddr = fir::BoxAddrOp::create(builder, loc, typedBox);
  return rawAddr;
}

static Value genDescriptorGetTotalElements(fir::FirOpBuilder &builder,
                                           Location loc, Value boxDesc) {
  Value box = boxDesc;
  if (auto refBox = dyn_cast<fir::ReferenceType>(boxDesc.getType())) {
    box = fir::LoadOp::create(builder, loc, boxDesc);
  }
  assert(isa<fir::BoxType>(box.getType()) &&
         "Unknown type passed to genDescriptorGetTotalElements");
  auto i64Type = builder.getI64Type();
  return fir::BoxTotalElementsOp::create(builder, loc, i64Type, box);
}

static Value genDescriptorGetEleSize(fir::FirOpBuilder &builder, Location loc,
                                     Value boxDesc) {
  Value box = boxDesc;
  if (auto refBox = dyn_cast<fir::ReferenceType>(boxDesc.getType())) {
    box = fir::LoadOp::create(builder, loc, boxDesc);
  }
  assert(isa<fir::BoxType>(box.getType()) &&
         "Unknown type passed to genDescriptorGetElementSize");
  auto i64Type = builder.getI64Type();
  return fir::BoxEleSizeOp::create(builder, loc, i64Type, box);
}

static Value genDescriptorGetDataSizeInBytes(fir::FirOpBuilder &builder,
                                             Location loc, Value boxDesc) {
  Value box = boxDesc;
  if (auto refBox = dyn_cast<fir::ReferenceType>(boxDesc.getType())) {
    box = fir::LoadOp::create(builder, loc, boxDesc);
  }
  assert(isa<fir::BoxType>(box.getType()) &&
         "Unknown type passed to genDescriptorGetElementSize");
  Value eleSize = genDescriptorGetEleSize(builder, loc, box);
  Value totalElements = genDescriptorGetTotalElements(builder, loc, box);
  return mlir::arith::MulIOp::create(builder, loc, totalElements, eleSize);
}

static mlir::Value genOmpGetMappedPtrIfPresent(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value hostPtr,
                                               mlir::Value deviceNum,
                                               mlir::ModuleOp module) {
  auto *context = builder.getContext();
  auto voidPtrType = fir::LLVMPointerType::get(context, builder.getI8Type());
  auto i32Type = builder.getI32Type();
  auto funcName = "omp_get_mapped_ptr";
  auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(funcName);

  if (!funcOp) {
    auto funcType =
        mlir::FunctionType::get(context, {voidPtrType, i32Type}, {voidPtrType});

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    funcOp = mlir::func::FuncOp::create(builder, loc, funcName, funcType);
    funcOp.setPrivate();
  }

  llvm::SmallVector<mlir::Value> args;
  args.push_back(fir::ConvertOp::create(builder, loc, voidPtrType, hostPtr));
  args.push_back(fir::ConvertOp::create(builder, loc, i32Type, deviceNum));
  auto callOp = fir::CallOp::create(builder, loc, funcOp, args);
  auto mappedPtr = callOp.getResult(0);
  auto isNull = builder.genIsNullAddr(loc, mappedPtr);
  auto convertedHostPtr =
      fir::ConvertOp::create(builder, loc, voidPtrType, hostPtr);
  auto result = arith::SelectOp::create(builder, loc, isNull, convertedHostPtr,
                                        mappedPtr);
  return result;
}

static void genOmpTargetMemcpyCall(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value dst,
                                   mlir::Value src, mlir::Value length,
                                   mlir::Value dstOffset, mlir::Value srcOffset,
                                   mlir::Value device, mlir::ModuleOp module) {
  auto *context = builder.getContext();
  // int omp_target_memcpy(void *dst, const void *src, size_t length,
  //                       size_t dst_offset, size_t src_offset,
  //                       int dst_device, int src_device)
  auto funcName = "omp_target_memcpy";
  auto voidPtrType = fir::LLVMPointerType::get(context, builder.getI8Type());
  auto sizeTType = builder.getI64Type(); // assuming size_t is 64-bit
  auto i32Type = builder.getI32Type();
  auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(funcName);

  if (!funcOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    llvm::SmallVector<mlir::Type> argTypes = {
        voidPtrType, voidPtrType, sizeTType, sizeTType,
        sizeTType,   i32Type,     i32Type};
    auto funcType = mlir::FunctionType::get(context, argTypes, {i32Type});
    funcOp = mlir::func::FuncOp::create(builder, loc, funcName, funcType);
    funcOp.setPrivate();
  }

  llvm::SmallVector<mlir::Value> args{dst,       src,    length, dstOffset,
                                      srcOffset, device, device};
  fir::CallOp::create(builder, loc, funcOp, args);
  return;
}

// moveToHost method clones all the ops from target region outside of it.
// It hoists runtime functions and replaces them with omp vesions.
// Also hoists and replaces fir.allocmem with omp.target_allocmem and
// fir.freemem with omp.target_freemem
static void moveToHost(omp::TargetOp targetOp, RewriterBase &rewriter,
                       mlir::ModuleOp module) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *targetBlock = &targetOp.getRegion().front();
  assert(targetBlock == &targetOp.getRegion().back());
  IRMapping mapping;
  for (unsigned i = 0; i < targetOp.getMapVars().size(); ++i) {
    Value mapInfo = targetOp.getMapVars()[i];
    BlockArgument arg = targetBlock->getArguments()[i];
    Operation *op = mapInfo.getDefiningOp();
    assert(op);
    auto mapInfoOp = cast<omp::MapInfoOp>(op);
    // map the block argument to the host-side variable pointer
    mapping.map(arg, mapInfoOp.getVarPtr());
  }
  unsigned mapSize = targetOp.getMapVars().size();
  for (unsigned i = 0; i < targetOp.getPrivateVars().size(); ++i) {
    Value privateVar = targetOp.getPrivateVars()[i];
    // The mapping should link the device-side variable to the host-side one.
    BlockArgument arg = targetBlock->getArguments()[mapSize + i];
    // Map the device-side copy (`arg`) to the host-side value (`privateVar`).
    mapping.map(arg, privateVar);
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
      if (isRuntimeCall(clonedOp)) {
        fir::CallOp runtimeCall = cast<fir::CallOp>(op);
        if ((*runtimeCall.getCallee()).getRootReference().getValue() ==
            "_FortranAAssign") {
          opsToReplace.push_back(clonedOp);
        } else {
          llvm_unreachable("Unhandled runtime call hoisting.");
        }
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
    } else if (isRuntimeCall(op)) {
      rewriter.setInsertionPoint(op);
      fir::CallOp runtimeCall = cast<fir::CallOp>(op);
      SmallVector<Value> operands = runtimeCall.getOperands();
      mlir::Location loc = runtimeCall.getLoc();
      fir::FirOpBuilder builder{rewriter, op};
      assert(operands.size() == 4);
      Value sourceFile{operands[2]}, sourceLine{operands[3]};

      auto fromBaseAddr =
          genDescriptorGetBaseAddress(builder, loc, operands[1]);
      auto toBaseAddr = genDescriptorGetBaseAddress(builder, loc, operands[0]);
      auto dataSizeInBytes =
          genDescriptorGetDataSizeInBytes(builder, loc, operands[1]);

      Value toPtr =
          genOmpGetMappedPtrIfPresent(builder, loc, toBaseAddr, device, module);
      Value fromPtr = genOmpGetMappedPtrIfPresent(builder, loc, fromBaseAddr,
                                                  device, module);
      Value zero = genI64Constant(loc, rewriter, 0);
      genOmpTargetMemcpyCall(builder, loc, toPtr, fromPtr, dataSizeInBytes,
                             zero, zero, device, module);
      rewriter.eraseOp(op);
    }
  }
  rewriter.eraseOp(targetOp);
}

void fissionTarget(omp::TargetOp targetOp, RewriterBase &rewriter,
                   mlir::ModuleOp module) {
  auto tuple = getNestedOpToIsolate(targetOp);
  if (!tuple) {
    LLVM_DEBUG(llvm::dbgs() << " No op to isolate\n");
    moveToHost(targetOp, rewriter, module);
    return;
  }

  Operation *toIsolate = std::get<0>(*tuple);
  bool splitBefore = !std::get<1>(*tuple);
  bool splitAfter = !std::get<2>(*tuple);

  if (splitBefore && splitAfter) {
    auto res = isolateOp(toIsolate, splitAfter, rewriter);
    moveToHost(res.preTargetOp, rewriter, module);
    fissionTarget(res.postTargetOp, rewriter, module);
    return;
  }
  if (splitBefore) {
    auto res = isolateOp(toIsolate, splitAfter, rewriter);
    moveToHost(res.preTargetOp, rewriter, module);
    return;
  }
  if (splitAfter) {
    auto res = isolateOp(toIsolate->getNextNode(), splitAfter, rewriter);
    fissionTarget(res.postTargetOp, rewriter, module);
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
          fissionTarget(res->targetOp, rewriter, moduleOp);
      }
    }
  }
};
} // namespace
