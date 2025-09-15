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
// runtime function "Assign" with omp.target_memcpy.
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
// operation nested in an omp.workdistribute region.
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
  // True if the op is a runtime call to Assign
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

// Generate omp.parallel operation with an empty region.
static void genParallelOp(Location loc, OpBuilder &rewriter, bool composite) {
  auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
  parallelOp.setComposite(composite);
  rewriter.createBlock(&parallelOp.getRegion());
  rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));
  return;
}

// Generate omp.distribute operation with an empty region.
static void genDistributeOp(Location loc, OpBuilder &rewriter, bool composite) {
  mlir::omp::DistributeOperands distributeClauseOps;
  auto distributeOp =
      rewriter.create<mlir::omp::DistributeOp>(loc, distributeClauseOps);
  distributeOp.setComposite(composite);
  auto distributeBlock = rewriter.createBlock(&distributeOp.getRegion());
  rewriter.setInsertionPointToStart(distributeBlock);
  return;
}

// Generate loop nest clause operands from fir.do_loop operation.
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

// Generate omp.wsloop operation with an empty region and
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

static bool
WorkdistributeDoLower(omp::WorkdistributeOp workdistribute,
                      SetVector<omp::TargetOp> &targetOpsToProcess) {
  OpBuilder rewriter(workdistribute);
  auto doLoop = getPerfectlyNested<fir::DoLoopOp>(workdistribute);
  auto wdLoc = workdistribute->getLoc();
  if (doLoop && shouldParallelize(doLoop)) {
    assert(doLoop.getReduceOperands().empty());

    // Record the target ops to process later
    if (auto teamsOp = dyn_cast<omp::TeamsOp>(workdistribute->getParentOp())) {
      auto targetOp = dyn_cast<omp::TargetOp>(teamsOp->getParentOp());
      if (targetOp) {
        targetOpsToProcess.insert(targetOp);
      }
    }
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
// If only the terminator remains in teams after hoisting, we erase teams op.
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

// If multiple workdistribute are nested in a target regions, we will need to
// split the target region, but we want to preserve the data semantics of the
// original data region and avoid unnecessary data movement at each of the
// subkernels - we split the target region into a target_data{target}
// nest where only the outer one moves the data
std::optional<omp::TargetOp> splitTargetData(omp::TargetOp targetOp,
                                             RewriterBase &rewriter) {
  auto loc = targetOp->getLoc();
  if (targetOp.getMapVars().empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << " target region has no data maps\n");
    return std::nullopt;
  }
  // Collect all the mapinfo ops
  SmallVector<omp::MapInfoOp> mapInfos;
  for (auto opr : targetOp.getMapVars()) {
    auto mapInfo = cast<omp::MapInfoOp>(opr.getDefiningOp());
    mapInfos.push_back(mapInfo);
  }

  rewriter.setInsertionPoint(targetOp);
  SmallVector<Value> innerMapInfos;
  SmallVector<Value> outerMapInfos;
  // Create new mapinfo ops for the inner target region
  for (auto mapInfo : mapInfos) {
    auto originalMapType =
        (llvm::omp::OpenMPOffloadMappingFlags)(mapInfo.getMapType());
    auto originalCaptureType = mapInfo.getMapCaptureType();
    llvm::omp::OpenMPOffloadMappingFlags newMapType;
    mlir::omp::VariableCaptureKind newCaptureType;
    // For bycopy, we keep the same map type and capture type
    // For byref, we change the map type to none and keep the capture type
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
  // Create the target data op
  auto targetDataOp = rewriter.create<omp::TargetDataOp>(
      loc, device, ifExpr, outerMapInfos, deviceAddrVars, devicePtrVars);
  auto taregtDataBlock = rewriter.createBlock(&targetDataOp.getRegion());
  rewriter.create<mlir::omp::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(taregtDataBlock);
  // Create the inner target op
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
  return newTargetOp;
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

    if (isa<omp::TeamsOp>(&op))
      return {{&op, first, last}};
  }
  return std::nullopt;
}

// Temporary structure to hold the two mapinfo ops
struct TempOmpVar {
  omp::MapInfoOp from, to;
};

// isPtr checks if the type is a pointer or reference type.
static bool isPtr(Type ty) {
  return isa<fir::ReferenceType>(ty) || isa<LLVM::LLVMPointerType>(ty);
}

// getPtrTypeForOmp returns an LLVM pointer type for the given type.
static Type getPtrTypeForOmp(Type ty) {
  if (isPtr(ty))
    return LLVM::LLVMPointerType::get(ty.getContext());
  else
    return fir::LLVMPointerType::get(ty);
}

// allocateTempOmpVar allocates a temporary variable for OpenMP mapping
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
}

// usedOutsideSplit checks if a value is used outside the split operation.
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

// isRecomputableAfterFission checks if an operation can be recomputed
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

// collectNonRecomputableDeps collects dependencies that cannot be recomputed
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

// createBlockArgsAndMap creates block arguments and maps them
static void createBlockArgsAndMap(Location loc, RewriterBase &rewriter,
                                  omp::TargetOp &targetOp, Block *targetBlock,
                                  Block *newTargetBlock,
                                  SmallVector<Value> &hostEvalVars,
                                  SmallVector<Value> &mapOperands,
                                  SmallVector<Value> &allocs,
                                  IRMapping &irMapping) {
  // FIRST: Map `host_eval_vars` to block arguments
  unsigned originalHostEvalVarsSize = targetOp.getHostEvalVars().size();
  for (unsigned i = 0; i < hostEvalVars.size(); ++i) {
    Value originalValue;
    BlockArgument newArg;
    if (i < originalHostEvalVarsSize) {
      originalValue = targetBlock->getArgument(i); // Host_eval args come first
      newArg = newTargetBlock->addArgument(originalValue.getType(),
                                           originalValue.getLoc());
    } else {
      originalValue = hostEvalVars[i];
      newArg = newTargetBlock->addArgument(originalValue.getType(),
                                           originalValue.getLoc());
    }
    irMapping.map(originalValue, newArg);
  }

  // SECOND: Map `map_operands` to block arguments
  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  for (unsigned i = 0; i < mapOperands.size(); ++i) {
    Value originalValue;
    BlockArgument newArg;
    // Map the new arguments from the original block.
    if (i < originalMapVarsSize) {
      originalValue = targetBlock->getArgument(originalHostEvalVarsSize +
                                               i); // Offset by host_eval count
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

  // THIRD: Map `private_vars` to block arguments (if any)
  unsigned originalPrivateVarsSize = targetOp.getPrivateVars().size();
  for (unsigned i = 0; i < originalPrivateVarsSize; ++i) {
    auto originalArg = targetBlock->getArgument(originalHostEvalVarsSize +
                                                originalMapVarsSize + i);
    auto newArg = newTargetBlock->addArgument(originalArg.getType(),
                                              originalArg.getLoc());
    irMapping.map(originalArg, newArg);
  }
  return;
}

// reloadCacheAndRecompute reloads cached values and recomputes operations
static void reloadCacheAndRecompute(
    Location loc, RewriterBase &rewriter, Operation *splitBefore,
    omp::TargetOp &targetOp, Block *targetBlock, Block *newTargetBlock,
    SmallVector<Value> &hostEvalVars, SmallVector<Value> &mapOperands,
    SmallVector<Value> &allocs, SetVector<Operation *> &toRecompute,
    IRMapping &irMapping) {
  // Handle the load operations for the allocs.
  rewriter.setInsertionPointToStart(newTargetBlock);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  unsigned hostEvalVarsSize = hostEvalVars.size();
  // Create Stores for allocs.
  for (unsigned i = 0; i < allocs.size(); ++i) {
    Value original = allocs[i];
    // Get the new block argument for this specific allocated value.
    Value newArg =
        newTargetBlock->getArgument(hostEvalVarsSize + originalMapVarsSize + i);
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
  // Clone the operations if they are in the toRecompute set.
  for (auto it = targetBlock->begin(); it != splitBefore->getIterator(); it++) {
    if (toRecompute.contains(&*it))
      rewriter.clone(*it, irMapping);
  }
}

// Given a teamsOp, navigate down the nested structure to find the
// innermost LoopNestOp. The expected nesting is:
// teams -> parallel -> distribute -> wsloop -> loop_nest
static mlir::omp::LoopNestOp getLoopNestFromTeams(mlir::omp::TeamsOp teamsOp) {
  if (teamsOp.getRegion().empty())
    return nullptr;
  // Ensure the teams region has a single block.
  if (teamsOp.getRegion().getBlocks().size() != 1)
    return nullptr;
  // Find parallel op inside teams
  mlir::omp::ParallelOp parallelOp = nullptr;
  for (auto &op : teamsOp.getRegion().front()) {
    if (auto parallel = dyn_cast<mlir::omp::ParallelOp>(op)) {
      parallelOp = parallel;
      break;
    }
  }
  if (!parallelOp)
    return nullptr;

  // Find distribute op inside parallel
  mlir::omp::DistributeOp distributeOp = nullptr;
  for (auto &op : parallelOp.getRegion().front()) {
    if (auto distribute = dyn_cast<mlir::omp::DistributeOp>(op)) {
      distributeOp = distribute;
      break;
    }
  }
  if (!distributeOp)
    return nullptr;

  // Find wsloop op inside distribute
  mlir::omp::WsloopOp wsloopOp = nullptr;
  for (auto &op : distributeOp.getRegion().front()) {
    if (auto wsloop = dyn_cast<mlir::omp::WsloopOp>(op)) {
      wsloopOp = wsloop;
      break;
    }
  }
  if (!wsloopOp)
    return nullptr;

  // Find loop_nest op inside wsloop
  for (auto &op : wsloopOp.getRegion().front()) {
    if (auto loopNest = dyn_cast<mlir::omp::LoopNestOp>(op)) {
      return loopNest;
    }
  }

  return nullptr;
}

// Generate LLVM constant operations for i32 and i64 types.
static mlir::LLVM::ConstantOp
genI32Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

// Generate LLVM constant operations for i64 type.
static mlir::LLVM::ConstantOp
genI64Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i64Ty = rewriter.getI64Type();
  mlir::IntegerAttr attr = rewriter.getI64IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, attr);
}

// Given a box descriptor, extract the base address of the data it describes.
// If the box descriptor is a reference, load it first.
// The base address is returned as an i8* pointer.
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

// Given a box descriptor, extract the total number of elements in the array it
// describes. If the box descriptor is a reference, load it first.
// The total number of elements is returned as an i64 value.
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

// Given a box descriptor, extract the size of each element in the array it
// describes. If the box descriptor is a reference, load it first.
// The element size is returned as an i64 value.
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

// Given a box descriptor, compute the total size in bytes of the data it
// describes. This is done by multiplying the total number of elements by the
// size of each element. If the box descriptor is a reference, load it first.
// The total size in bytes is returned as an i64 value.
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

// Generate a call to the OpenMP runtime function `omp_get_mapped_ptr` to
// retrieve the device pointer corresponding to a given host pointer and device
// number. If no mapping exists, the original host pointer is returned.
// Signature:
//   void *omp_get_mapped_ptr(void *host_ptr, int device_num);
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

// Generate a call to the OpenMP runtime function `omp_target_memcpy` to
// perform memory copy between host and device or between devices.
// Signature:
//   int omp_target_memcpy(void *dst, const void *src, size_t length,
//                         size_t dst_offset, size_t src_offset,
//                         int dst_device, int src_device);
static void genOmpTargetMemcpyCall(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value dst,
                                   mlir::Value src, mlir::Value length,
                                   mlir::Value dstOffset, mlir::Value srcOffset,
                                   mlir::Value device, mlir::ModuleOp module) {
  auto *context = builder.getContext();
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

// Struct to hold the host eval vars corresponding to loop bounds and steps
struct HostEvalVars {
  SmallVector<Value> lbs;
  SmallVector<Value> ubs;
  SmallVector<Value> steps;
};

// moveToHost method clones all the ops from target region outside of it.
// It hoists runtime functions and replaces them with omp vesions.
// Also hoists and replaces fir.allocmem with omp.target_allocmem and
// fir.freemem with omp.target_freemem
static void moveToHost(omp::TargetOp targetOp, RewriterBase &rewriter,
                       mlir::ModuleOp module,
                       struct HostEvalVars &hostEvalVars) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block *targetBlock = &targetOp.getRegion().front();
  assert(targetBlock == &targetOp.getRegion().back());
  IRMapping mapping;
  // create mapping for host_eval_vars
  unsigned hostEvalVarCount = targetOp.getHostEvalVars().size();
  for (unsigned i = 0; i < targetOp.getHostEvalVars().size(); ++i) {
    Value hostEvalVar = targetOp.getHostEvalVars()[i];
    BlockArgument arg = targetBlock->getArguments()[i];
    mapping.map(arg, hostEvalVar);
  }
  // create mapping for map_vars
  for (unsigned i = 0; i < targetOp.getMapVars().size(); ++i) {
    Value mapInfo = targetOp.getMapVars()[i];
    BlockArgument arg = targetBlock->getArguments()[hostEvalVarCount + i];
    Operation *op = mapInfo.getDefiningOp();
    assert(op);
    auto mapInfoOp = cast<omp::MapInfoOp>(op);
    // map the block argument to the host-side variable pointer
    mapping.map(arg, mapInfoOp.getVarPtr());
  }
  // create mapping for private_vars
  unsigned mapSize = targetOp.getMapVars().size();
  for (unsigned i = 0; i < targetOp.getPrivateVars().size(); ++i) {
    Value privateVar = targetOp.getPrivateVars()[i];
    // The mapping should link the device-side variable to the host-side one.
    BlockArgument arg =
        targetBlock->getArguments()[hostEvalVarCount + mapSize + i];
    // Map the device-side copy (`arg`) to the host-side value (`privateVar`).
    mapping.map(arg, privateVar);
  }

  rewriter.setInsertionPoint(targetOp);
  SmallVector<Operation *> opsToReplace;
  Value device = targetOp.getDevice();

  if (!device) {
    device = genI32Constant(targetOp.getLoc(), rewriter, 0);
  }
  // Clone all operations.
  for (auto it = targetBlock->begin(), end = std::prev(targetBlock->end());
       it != end; ++it) {
    auto *op = &*it;
    Operation *clonedOp = rewriter.clone(*op, mapping);
    // Map the results of the original op to the cloned op.
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
    // Collect the ops to be replaced.
    if (isa<fir::AllocMemOp>(clonedOp) || isa<fir::FreeMemOp>(clonedOp))
      opsToReplace.push_back(clonedOp);
    // Check for runtime calls to be replaced.
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
  // Replace fir.allocmem with omp.target_allocmem.
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
    }
    // Replace fir.freemem with omp.target_freemem.
    else if (auto freeOp = dyn_cast<fir::FreeMemOp>(op)) {
      rewriter.setInsertionPoint(freeOp);
      auto firConvertOp = rewriter.create<fir::ConvertOp>(
          freeOp.getLoc(), rewriter.getI64Type(), freeOp.getHeapref());
      rewriter.create<omp::TargetFreeMemOp>(freeOp.getLoc(), device,
                                            firConvertOp.getResult());
      rewriter.eraseOp(freeOp);
    }
    // fir.declare changes its type when hoisting it out of omp.target to
    // omp.target_data Introduce a load, if original declareOp input is not of
    // reference type, but cloned delcareOp input is reference type.
    else if (fir::DeclareOp clonedDeclareOp = dyn_cast<fir::DeclareOp>(op)) {
      Type clonedInType = clonedDeclareOp.getMemref().getType();
      fir::ReferenceType clonedRefType =
          dyn_cast<fir::ReferenceType>(clonedInType);
      Type clonedEleTy = clonedRefType.getElementType();
      rewriter.setInsertionPoint(op);
      Value loadedValue = rewriter.create<fir::LoadOp>(
          clonedDeclareOp.getLoc(), clonedEleTy, clonedDeclareOp.getMemref());
      clonedDeclareOp.getResult().replaceAllUsesWith(loadedValue);
    }
    // Replace runtime calls with omp versions.
    else if (isRuntimeCall(op)) {
      rewriter.setInsertionPoint(op);
      fir::CallOp runtimeCall = cast<fir::CallOp>(op);
      SmallVector<Value> operands = runtimeCall.getOperands();
      mlir::Location loc = runtimeCall.getLoc();
      fir::FirOpBuilder builder{rewriter, op};
      assert(operands.size() == 4);
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

  // Update the host_eval_vars to use the mapped values.
  for (size_t i = 0; i < hostEvalVars.lbs.size(); ++i) {
    hostEvalVars.lbs[i] = mapping.lookup(hostEvalVars.lbs[i]);
    hostEvalVars.ubs[i] = mapping.lookup(hostEvalVars.ubs[i]);
    hostEvalVars.steps[i] = mapping.lookup(hostEvalVars.steps[i]);
  }
  // Finally erase the original targetOp.
  rewriter.eraseOp(targetOp);
}

// Result of isolateOp method
struct SplitResult {
  omp::TargetOp preTargetOp;
  omp::TargetOp isolatedTargetOp;
  omp::TargetOp postTargetOp;
};

// computeAllocsCacheRecomputable method computes the allocs needed to cache
// the values that are used outside the split point. It also computes the ops
// that need to be cached and the ops that can be recomputed after the split.
static void computeAllocsCacheRecomputable(
    omp::TargetOp targetOp, Operation *splitBeforeOp, RewriterBase &rewriter,
    SmallVector<Value> &preMapOperands, SmallVector<Value> &postMapOperands,
    SmallVector<Value> &allocs, SmallVector<Value> &requiredVals,
    SetVector<Operation *> &nonRecomputable, SetVector<Operation *> &toCache,
    SetVector<Operation *> &toRecompute) {
  auto *targetBlock = &targetOp.getRegion().front();
  // Find all values that are used outside the split point.
  for (auto it = targetBlock->begin(); it != splitBeforeOp->getIterator();
       it++) {
    // Check if any of the results are used outside the split point.
    for (auto res : it->getResults()) {
      if (usedOutsideSplit(res, splitBeforeOp))
        requiredVals.push_back(res);
    }
    // If the op is not recomputable, add it to the nonRecomputable set.
    if (!isRecomputableAfterFission(&*it, splitBeforeOp))
      nonRecomputable.insert(&*it);
  }
  // For each required value, collect its dependencies.
  for (auto requiredVal : requiredVals)
    collectNonRecomputableDeps(requiredVal, targetOp, nonRecomputable, toCache,
                               toRecompute);
  // For each op in toCache, create an alloc and update the pre and post map
  // operands.
  for (Operation *op : toCache) {
    for (auto res : op->getResults()) {
      auto alloc =
          allocateTempOmpVar(targetOp.getLoc(), res.getType(), rewriter);
      allocs.push_back(res);
      preMapOperands.push_back(alloc.from);
      postMapOperands.push_back(alloc.to);
    }
  }
}

// genPreTargetOp method generates the preTargetOp that contains all the ops
// before the split point. It also creates the block arguments and maps the
// values accordingly. It also creates the store operations for the allocs.
static omp::TargetOp
genPreTargetOp(omp::TargetOp targetOp, SmallVector<Value> &preMapOperands,
               SmallVector<Value> &allocs, Operation *splitBeforeOp,
               RewriterBase &rewriter, struct HostEvalVars &hostEvalVars,
               bool isTargetDevice) {
  auto loc = targetOp.getLoc();
  auto *targetBlock = &targetOp.getRegion().front();
  SmallVector<Value> preHostEvalVars{targetOp.getHostEvalVars()};
  // update the hostEvalVars of preTargetOp
  omp::TargetOp preTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(), preHostEvalVars,
      targetOp.getIfExpr(), targetOp.getInReductionVars(),
      targetOp.getInReductionByrefAttr(), targetOp.getInReductionSymsAttr(),
      targetOp.getIsDevicePtrVars(), preMapOperands, targetOp.getNowaitAttr(),
      targetOp.getPrivateVars(), targetOp.getPrivateSymsAttr(),
      targetOp.getPrivateNeedsBarrierAttr(), targetOp.getThreadLimit(),
      targetOp.getPrivateMapsAttr());
  auto *preTargetBlock = rewriter.createBlock(
      &preTargetOp.getRegion(), preTargetOp.getRegion().begin(), {}, {});
  IRMapping preMapping;
  // Create block arguments and map the values.
  createBlockArgsAndMap(loc, rewriter, targetOp, targetBlock, preTargetBlock,
                        preHostEvalVars, preMapOperands, allocs, preMapping);

  // Handle the store operations for the allocs.
  rewriter.setInsertionPointToStart(preTargetBlock);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

  // Clone the original operations.
  for (auto it = targetBlock->begin(); it != splitBeforeOp->getIterator();
       it++) {
    rewriter.clone(*it, preMapping);
  }

  unsigned originalHostEvalVarsSize = preHostEvalVars.size();
  unsigned originalMapVarsSize = targetOp.getMapVars().size();
  // Create Stores for allocs.
  for (unsigned i = 0; i < allocs.size(); ++i) {
    Value originalResult = allocs[i];
    Value toStore = preMapping.lookup(originalResult);
    // Get the new block argument for this specific allocated value.
    Value newArg = preTargetBlock->getArgument(originalHostEvalVarsSize +
                                               originalMapVarsSize + i);
    // Create the store operation.
    if (isPtr(originalResult.getType())) {
      if (!isa<LLVM::LLVMPointerType>(toStore.getType()))
        toStore = rewriter.create<fir::ConvertOp>(loc, llvmPtrTy, toStore);
      rewriter.create<LLVM::StoreOp>(loc, toStore, newArg);
    } else {
      rewriter.create<fir::StoreOp>(loc, toStore, newArg);
    }
  }
  rewriter.create<omp::TerminatorOp>(loc);

  // Update hostEvalVars with the mapped values for the loop bounds if we have
  // a loopNestOp and we are not generating code for the target device.
  omp::LoopNestOp loopNestOp =
      getLoopNestFromTeams(cast<omp::TeamsOp>(splitBeforeOp));
  if (loopNestOp && !isTargetDevice) {
    for (size_t i = 0; i < loopNestOp.getLoopLowerBounds().size(); ++i) {
      Value lb = loopNestOp.getLoopLowerBounds()[i];
      Value ub = loopNestOp.getLoopUpperBounds()[i];
      Value step = loopNestOp.getLoopSteps()[i];

      hostEvalVars.lbs.push_back(preMapping.lookup(lb));
      hostEvalVars.ubs.push_back(preMapping.lookup(ub));
      hostEvalVars.steps.push_back(preMapping.lookup(step));
    }
  }

  return preTargetOp;
}

// genIsolatedTargetOp method generates the isolatedTargetOp that contains the
// ops between the split point. It also creates the block arguments and maps
// the values accordingly. It also creates the load operations for the allocs
// and recomputes the necessary ops.
static omp::TargetOp
genIsolatedTargetOp(omp::TargetOp targetOp, SmallVector<Value> &postMapOperands,
                    Operation *splitBeforeOp, RewriterBase &rewriter,
                    SmallVector<Value> &allocs,
                    SetVector<Operation *> &toRecompute,
                    struct HostEvalVars &hostEvalVars, bool isTargetDevice) {
  auto loc = targetOp.getLoc();
  auto *targetBlock = &targetOp.getRegion().front();
  SmallVector<Value> isolatedHostEvalVars{targetOp.getHostEvalVars()};
  // update the hostEvalVars of isolatedTargetOp
  if (!hostEvalVars.lbs.empty() && !isTargetDevice) {
    for (size_t i = 0; i < hostEvalVars.lbs.size(); ++i) {
      isolatedHostEvalVars.push_back(hostEvalVars.lbs[i]);
    }
    for (size_t i = 0; i < hostEvalVars.ubs.size(); ++i) {
      isolatedHostEvalVars.push_back(hostEvalVars.ubs[i]);
    }
    for (size_t i = 0; i < hostEvalVars.steps.size(); ++i) {
      isolatedHostEvalVars.push_back(hostEvalVars.steps[i]);
    }
  }
  // Create the isolated target op
  omp::TargetOp isolatedTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
      isolatedHostEvalVars, targetOp.getIfExpr(), targetOp.getInReductionVars(),
      targetOp.getInReductionByrefAttr(), targetOp.getInReductionSymsAttr(),
      targetOp.getIsDevicePtrVars(), postMapOperands, targetOp.getNowaitAttr(),
      targetOp.getPrivateVars(), targetOp.getPrivateSymsAttr(),
      targetOp.getPrivateNeedsBarrierAttr(), targetOp.getThreadLimit(),
      targetOp.getPrivateMapsAttr());
  auto *isolatedTargetBlock =
      rewriter.createBlock(&isolatedTargetOp.getRegion(),
                           isolatedTargetOp.getRegion().begin(), {}, {});
  IRMapping isolatedMapping;
  // Create block arguments and map the values.
  createBlockArgsAndMap(loc, rewriter, targetOp, targetBlock,
                        isolatedTargetBlock, isolatedHostEvalVars,
                        postMapOperands, allocs, isolatedMapping);
  // Handle the load operations for the allocs and recompute ops.
  reloadCacheAndRecompute(loc, rewriter, splitBeforeOp, targetOp, targetBlock,
                          isolatedTargetBlock, isolatedHostEvalVars,
                          postMapOperands, allocs, toRecompute,
                          isolatedMapping);

  // Clone the original operations.
  rewriter.clone(*splitBeforeOp, isolatedMapping);
  rewriter.create<omp::TerminatorOp>(loc);

  // update the loop bounds in the isolatedTargetOp if we have host_eval vars
  // and we are not generating code for the target device.
  if (!hostEvalVars.lbs.empty() && !isTargetDevice) {
    omp::TeamsOp teamsOp;
    for (auto &op : *isolatedTargetBlock) {
      if (isa<omp::TeamsOp>(&op))
        teamsOp = cast<omp::TeamsOp>(&op);
    }
    assert(teamsOp && "No teamsOp found in isolated target region");
    // Get the loopNestOp inside the teamsOp
    auto loopNestOp = getLoopNestFromTeams(teamsOp);
    // Get the BlockArgs related to host_eval vars and update loop_nest bounds
    // to them
    unsigned originalHostEvalVarsSize = targetOp.getHostEvalVars().size();
    unsigned index = originalHostEvalVarsSize;
    // Replace loop bounds with the block arguments passed down via host_eval
    SmallVector<Value> lbs, ubs, steps;

    // Collect new lb/ub/step values from target block args
    for (size_t i = 0; i < hostEvalVars.lbs.size(); ++i)
      lbs.push_back(isolatedTargetBlock->getArgument(index++));

    for (size_t i = 0; i < hostEvalVars.ubs.size(); ++i)
      ubs.push_back(isolatedTargetBlock->getArgument(index++));

    for (size_t i = 0; i < hostEvalVars.steps.size(); ++i)
      steps.push_back(isolatedTargetBlock->getArgument(index++));

    // Reset the loop bounds
    loopNestOp.getLoopLowerBoundsMutable().assign(lbs);
    loopNestOp.getLoopUpperBoundsMutable().assign(ubs);
    loopNestOp.getLoopStepsMutable().assign(steps);
  }

  return isolatedTargetOp;
}

// genPostTargetOp method generates the postTargetOp that contains all the ops
// after the split point. It also creates the block arguments and maps the
// values accordingly. It also creates the load operations for the allocs
// and recomputes the necessary ops.
static omp::TargetOp genPostTargetOp(omp::TargetOp targetOp,
                                     Operation *splitBeforeOp,
                                     SmallVector<Value> &postMapOperands,
                                     RewriterBase &rewriter,
                                     SmallVector<Value> &allocs,
                                     SetVector<Operation *> &toRecompute) {
  auto loc = targetOp.getLoc();
  auto *targetBlock = &targetOp.getRegion().front();
  SmallVector<Value> postHostEvalVars{targetOp.getHostEvalVars()};
  // Create the post target op
  omp::TargetOp postTargetOp = rewriter.create<omp::TargetOp>(
      targetOp.getLoc(), targetOp.getAllocateVars(),
      targetOp.getAllocatorVars(), targetOp.getBareAttr(),
      targetOp.getDependKindsAttr(), targetOp.getDependVars(),
      targetOp.getDevice(), targetOp.getHasDeviceAddrVars(), postHostEvalVars,
      targetOp.getIfExpr(), targetOp.getInReductionVars(),
      targetOp.getInReductionByrefAttr(), targetOp.getInReductionSymsAttr(),
      targetOp.getIsDevicePtrVars(), postMapOperands, targetOp.getNowaitAttr(),
      targetOp.getPrivateVars(), targetOp.getPrivateSymsAttr(),
      targetOp.getPrivateNeedsBarrierAttr(), targetOp.getThreadLimit(),
      targetOp.getPrivateMapsAttr());
  // Create the block for postTargetOp
  auto *postTargetBlock = rewriter.createBlock(
      &postTargetOp.getRegion(), postTargetOp.getRegion().begin(), {}, {});
  IRMapping postMapping;
  // Create block arguments and map the values.
  createBlockArgsAndMap(loc, rewriter, targetOp, targetBlock, postTargetBlock,
                        postHostEvalVars, postMapOperands, allocs, postMapping);
  // Handle the load operations for the allocs and recompute ops.
  reloadCacheAndRecompute(loc, rewriter, splitBeforeOp, targetOp, targetBlock,
                          postTargetBlock, postHostEvalVars, postMapOperands,
                          allocs, toRecompute, postMapping);
  assert(splitBeforeOp->getNumResults() == 0 ||
         llvm::all_of(splitBeforeOp->getResults(),
                      [](Value result) { return result.use_empty(); }));
  // Clone the original operations after the split point.
  for (auto it = std::next(splitBeforeOp->getIterator());
       it != targetBlock->end(); it++)
    rewriter.clone(*it, postMapping);
  return postTargetOp;
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
// It also handles the mapping of variables and the caching/recomputing
// of values as needed.
static SplitResult isolateOp(Operation *splitBeforeOp, bool splitAfter,
                             RewriterBase &rewriter, mlir::ModuleOp module,
                             bool isTargetDevice) {
  auto targetOp = cast<omp::TargetOp>(splitBeforeOp->getParentOp());
  assert(targetOp);
  rewriter.setInsertionPoint(targetOp);

  // Prepare the map operands for preTargetOp and postTargetOp
  auto preMapOperands = SmallVector<Value>(targetOp.getMapVars());
  auto postMapOperands = SmallVector<Value>(targetOp.getMapVars());

  // Vectors to hold analysis results
  SmallVector<Value> requiredVals;
  SetVector<Operation *> toCache;
  SetVector<Operation *> toRecompute;
  SetVector<Operation *> nonRecomputable;
  SmallVector<Value> allocs;
  struct HostEvalVars hostEvalVars;

  // Analyze the ops in target region to determine which ops need to be
  // cached and which ops need to be recomputed
  computeAllocsCacheRecomputable(
      targetOp, splitBeforeOp, rewriter, preMapOperands, postMapOperands,
      allocs, requiredVals, nonRecomputable, toCache, toRecompute);

  rewriter.setInsertionPoint(targetOp);

  // Generate the preTargetOp that contains all the ops before splitBeforeOp.
  auto preTargetOp =
      genPreTargetOp(targetOp, preMapOperands, allocs, splitBeforeOp, rewriter,
                     hostEvalVars, isTargetDevice);

  // Move the ops of preTarget to host.
  moveToHost(preTargetOp, rewriter, module, hostEvalVars);
  rewriter.setInsertionPoint(targetOp);

  // Generate the isolatedTargetOp
  omp::TargetOp isolatedTargetOp =
      genIsolatedTargetOp(targetOp, postMapOperands, splitBeforeOp, rewriter,
                          allocs, toRecompute, hostEvalVars, isTargetDevice);

  omp::TargetOp postTargetOp = nullptr;
  // Generate the postTargetOp that contains all the ops after splitBeforeOp.
  if (splitAfter) {
    rewriter.setInsertionPoint(targetOp);
    postTargetOp = genPostTargetOp(targetOp, splitBeforeOp, postMapOperands,
                                   rewriter, allocs, toRecompute);
  }
  // Finally erase the original targetOp.
  rewriter.eraseOp(targetOp);
  return SplitResult{preTargetOp, isolatedTargetOp, postTargetOp};
}

// Recursively fission target ops until no more nested ops can be isolated.
static void fissionTarget(omp::TargetOp targetOp, RewriterBase &rewriter,
                          mlir::ModuleOp module, bool isTargetDevice) {
  auto tuple = getNestedOpToIsolate(targetOp);
  if (!tuple) {
    LLVM_DEBUG(llvm::dbgs() << " No op to isolate\n");
    struct HostEvalVars hostEvalVars;
    moveToHost(targetOp, rewriter, module, hostEvalVars);
    return;
  }
  Operation *toIsolate = std::get<0>(*tuple);
  bool splitBefore = !std::get<1>(*tuple);
  bool splitAfter = !std::get<2>(*tuple);

  if (splitBefore && splitAfter) {
    auto res =
        isolateOp(toIsolate, splitAfter, rewriter, module, isTargetDevice);
    fissionTarget(res.postTargetOp, rewriter, module, isTargetDevice);
    return;
  } else {
    llvm::errs() << "Unhandled case in fissionTarget\n";
    llvm::report_fatal_error("Unhandled case in fissionTarget");
  }
}

// Pass to lower omp.workdistribute ops.
class LowerWorkdistributePass
    : public flangomp::impl::LowerWorkdistributeBase<LowerWorkdistributePass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    auto moduleOp = getOperation();
    bool changed = false;
    SetVector<omp::TargetOp> targetOpsToProcess;
    moduleOp->walk([&](mlir::omp::WorkdistributeOp workdistribute) {
      changed |= FissionWorkdistribute(workdistribute);
    });
    moduleOp->walk([&](mlir::omp::WorkdistributeOp workdistribute) {
      changed |= WorkdistributeDoLower(workdistribute, targetOpsToProcess);
    });
    moduleOp->walk([&](mlir::omp::TeamsOp teams) {
      changed |= TeamsWorkdistributeToSingleOp(teams);
    });

    if (changed) {
      bool isTargetDevice =
          llvm::cast<mlir::omp::OffloadModuleInterface>(*moduleOp)
              .getIsTargetDevice();
      IRRewriter rewriter(&context);
      for (auto targetOp : targetOpsToProcess) {
        auto res = splitTargetData(targetOp, rewriter);
        if (res)
          fissionTarget(*res, rewriter, moduleOp, isTargetDevice);
      }
    }
  }
};
} // namespace
