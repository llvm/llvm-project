//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of omp.workshare to other omp constructs.
//
// This pass is tasked with parallelizing the loops nested in
// workshare.loop_wrapper while both the Fortran to mlir lowering and the hlfir
// to fir lowering pipelines are responsible for emitting the
// workshare.loop_wrapper ops where appropriate according to the
// `shouldUseWorkshareLowering` function.
//
//===----------------------------------------------------------------------===//

#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Dialect/FIROps.h>
#include <flang/Optimizer/Dialect/FIRType.h>
#include <flang/Optimizer/HLFIR/HLFIROps.h>
#include <flang/Optimizer/OpenMP/Passes.h>
#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKSHARE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workshare"

using namespace mlir;

namespace flangomp {

// Checks for nesting pattern below as we need to avoid sharing the work of
// statements which are nested in some constructs such as omp.critical or
// another omp.parallel.
//
// omp.workshare { // `wsOp`
//   ...
//     omp.T { // `parent`
//       ...
//         `op`
//
template <typename T>
static bool isNestedIn(omp::WorkshareOp wsOp, Operation *op) {
  T parent = op->getParentOfType<T>();
  if (!parent)
    return false;
  return wsOp->isProperAncestor(parent);
}

bool shouldUseWorkshareLowering(Operation *op) {
  auto parentWorkshare = op->getParentOfType<omp::WorkshareOp>();

  if (!parentWorkshare)
    return false;

  if (isNestedIn<omp::CriticalOp>(parentWorkshare, op))
    return false;

  // 2.8.3  workshare Construct
  // For a parallel construct, the construct is a unit of work with respect to
  // the workshare construct. The statements contained in the parallel construct
  // are executed by a new thread team.
  if (isNestedIn<omp::ParallelOp>(parentWorkshare, op))
    return false;

  // 2.8.2  single Construct
  // Binding The binding thread set for a single region is the current team. A
  // single region binds to the innermost enclosing parallel region.
  // Description Only one of the encountering threads will execute the
  // structured block associated with the single construct.
  if (isNestedIn<omp::SingleOp>(parentWorkshare, op))
    return false;

  // Do not use workshare lowering until we support CFG in omp.workshare
  if (parentWorkshare.getRegion().getBlocks().size() != 1)
    return false;

  return true;
}

} // namespace flangomp

namespace {

struct SingleRegion {
  Block::iterator begin, end;
};

static bool mustParallelizeOp(Operation *op) {
  return op
      ->walk([&](Operation *nested) {
        // We need to be careful not to pick up workshare.loop_wrapper in nested
        // omp.parallel{omp.workshare} regions, i.e. make sure that `nested`
        // binds to the workshare region we are currently handling.
        //
        // For example:
        //
        // omp.parallel {
        //   omp.workshare { // currently handling this
        //     omp.parallel {
        //       omp.workshare { // nested workshare
        //         omp.workshare.loop_wrapper {}
        //
        // Therefore, we skip if we encounter a nested omp.workshare.
        if (isa<omp::WorkshareOp>(nested))
          return WalkResult::skip();
        if (isa<omp::WorkshareLoopWrapperOp>(nested))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool isSafeToParallelize(Operation *op) {
  return isa<hlfir::DeclareOp>(op) || isa<fir::DeclareOp>(op) ||
         isMemoryEffectFree(op);
}

/// Simple shallow copies suffice for our purposes in this pass, so we implement
/// this simpler alternative to the full fledged `createCopyFunc` in the
/// frontend
static mlir::func::FuncOp createCopyFunc(mlir::Location loc, mlir::Type varType,
                                         fir::FirOpBuilder builder) {
  mlir::ModuleOp module = builder.getModule();
  auto rt = cast<fir::ReferenceType>(varType);
  mlir::Type eleTy = rt.getEleTy();
  std::string copyFuncName =
      fir::getTypeAsString(eleTy, builder.getKindMap(), "_workshare_copy");

  if (auto decl = module.lookupSymbol<mlir::func::FuncOp>(copyFuncName))
    return decl;

  // create function
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  llvm::SmallVector<mlir::Type> argsTy = {varType, varType};
  auto funcType = mlir::FunctionType::get(builder.getContext(), argsTy, {});
  mlir::func::FuncOp funcOp =
      modBuilder.create<mlir::func::FuncOp>(loc, copyFuncName, funcType);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  fir::factory::setInternalLinkage(funcOp);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      {loc, loc});
  builder.setInsertionPointToStart(&funcOp.getRegion().back());

  Value loaded = builder.create<fir::LoadOp>(loc, funcOp.getArgument(1));
  builder.create<fir::StoreOp>(loc, loaded, funcOp.getArgument(0));

  builder.create<mlir::func::ReturnOp>(loc);
  return funcOp;
}

static bool isUserOutsideSR(Operation *user, Operation *parentOp,
                            SingleRegion sr) {
  while (user->getParentOp() != parentOp)
    user = user->getParentOp();
  return sr.begin->getBlock() != user->getBlock() ||
         !(user->isBeforeInBlock(&*sr.end) && sr.begin->isBeforeInBlock(user));
}

static bool isTransitivelyUsedOutside(Value v, SingleRegion sr) {
  Block *srBlock = sr.begin->getBlock();
  Operation *parentOp = srBlock->getParentOp();

  for (auto &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (isUserOutsideSR(user, parentOp, sr))
      return true;

    // Now we know user is inside `sr`.

    // Results of nested users cannot be used outside of `sr`.
    if (user->getBlock() != srBlock)
      continue;

    // A non-safe to parallelize operation will be checked for uses outside
    // separately.
    if (!isSafeToParallelize(user))
      continue;

    // For safe to parallelize operations, we need to check if there is a
    // transitive use of `v` through them.
    for (auto res : user->getResults())
      if (isTransitivelyUsedOutside(res, sr))
        return true;
  }
  return false;
}

/// We clone pure operations in both the parallel and single blocks. this
/// functions cleans them up if they end up with no uses
static void cleanupBlock(Block *block) {
  for (Operation &op : llvm::make_early_inc_range(
           llvm::make_range(block->rbegin(), block->rend())))
    if (isOpTriviallyDead(&op))
      op.erase();
}

static void parallelizeRegion(Region &sourceRegion, Region &targetRegion,
                              IRMapping &rootMapping, Location loc,
                              mlir::DominanceInfo &di) {
  OpBuilder rootBuilder(sourceRegion.getContext());
  ModuleOp m = sourceRegion.getParentOfType<ModuleOp>();
  OpBuilder copyFuncBuilder(m.getBodyRegion());
  fir::FirOpBuilder firCopyFuncBuilder(copyFuncBuilder, m);

  auto mapReloadedValue =
      [&](Value v, OpBuilder allocaBuilder, OpBuilder singleBuilder,
          OpBuilder parallelBuilder, IRMapping singleMapping) -> Value {
    if (auto reloaded = rootMapping.lookupOrNull(v))
      return nullptr;
    Type ty = v.getType();
    Value alloc = allocaBuilder.create<fir::AllocaOp>(loc, ty);
    singleBuilder.create<fir::StoreOp>(loc, singleMapping.lookup(v), alloc);
    Value reloaded = parallelBuilder.create<fir::LoadOp>(loc, ty, alloc);
    rootMapping.map(v, reloaded);
    return alloc;
  };

  auto moveToSingle =
      [&](SingleRegion sr, OpBuilder allocaBuilder, OpBuilder singleBuilder,
          OpBuilder parallelBuilder) -> std::pair<bool, SmallVector<Value>> {
    IRMapping singleMapping = rootMapping;
    SmallVector<Value> copyPrivate;
    bool allParallelized = true;

    for (Operation &op : llvm::make_range(sr.begin, sr.end)) {
      if (isSafeToParallelize(&op)) {
        singleBuilder.clone(op, singleMapping);
        if (llvm::all_of(op.getOperands(), [&](Value opr) {
              // Either we have already remapped it
              bool remapped = rootMapping.contains(opr);
              // Or it is available because it dominates `sr`
              bool dominates = di.properlyDominates(opr, &*sr.begin);
              return remapped || dominates;
            })) {
          // Safe to parallelize operations which have all operands available in
          // the root parallel block can be executed there.
          parallelBuilder.clone(op, rootMapping);
        } else {
          // If any operand was not available, it means that there was no
          // transitive use of a non-safe-to-parallelize operation outside `sr`.
          // This means that there should be no transitive uses outside `sr` of
          // `op`.
          assert(llvm::all_of(op.getResults(), [&](Value v) {
            return !isTransitivelyUsedOutside(v, sr);
          }));
          allParallelized = false;
        }
      } else if (auto alloca = dyn_cast<fir::AllocaOp>(&op)) {
        auto hoisted =
            cast<fir::AllocaOp>(allocaBuilder.clone(*alloca, singleMapping));
        rootMapping.map(&*alloca, &*hoisted);
        rootMapping.map(alloca.getResult(), hoisted.getResult());
        copyPrivate.push_back(hoisted);
        allParallelized = false;
      } else {
        singleBuilder.clone(op, singleMapping);
        // Prepare reloaded values for results of operations that cannot be
        // safely parallelized and which are used after the region `sr`.
        for (auto res : op.getResults()) {
          if (isTransitivelyUsedOutside(res, sr)) {
            auto alloc = mapReloadedValue(res, allocaBuilder, singleBuilder,
                                          parallelBuilder, singleMapping);
            if (alloc)
              copyPrivate.push_back(alloc);
          }
        }
        allParallelized = false;
      }
    }
    singleBuilder.create<omp::TerminatorOp>(loc);
    return {allParallelized, copyPrivate};
  };

  for (Block &block : sourceRegion) {
    Block *targetBlock = rootBuilder.createBlock(
        &targetRegion, {}, block.getArgumentTypes(),
        llvm::map_to_vector(block.getArguments(),
                            [](BlockArgument arg) { return arg.getLoc(); }));
    rootMapping.map(&block, targetBlock);
    rootMapping.map(block.getArguments(), targetBlock->getArguments());
  }

  auto handleOneBlock = [&](Block &block) {
    Block &targetBlock = *rootMapping.lookup(&block);
    rootBuilder.setInsertionPointToStart(&targetBlock);
    Operation *terminator = block.getTerminator();
    SmallVector<std::variant<SingleRegion, Operation *>> regions;

    auto it = block.begin();
    auto getOneRegion = [&]() {
      if (&*it == terminator)
        return false;
      if (mustParallelizeOp(&*it)) {
        regions.push_back(&*it);
        it++;
        return true;
      }
      SingleRegion sr;
      sr.begin = it;
      while (&*it != terminator && !mustParallelizeOp(&*it))
        it++;
      sr.end = it;
      assert(sr.begin != sr.end);
      regions.push_back(sr);
      return true;
    };
    while (getOneRegion())
      ;

    for (auto [i, opOrSingle] : llvm::enumerate(regions)) {
      bool isLast = i + 1 == regions.size();
      if (std::holds_alternative<SingleRegion>(opOrSingle)) {
        OpBuilder singleBuilder(sourceRegion.getContext());
        Block *singleBlock = new Block();
        singleBuilder.setInsertionPointToStart(singleBlock);

        OpBuilder allocaBuilder(sourceRegion.getContext());
        Block *allocaBlock = new Block();
        allocaBuilder.setInsertionPointToStart(allocaBlock);

        OpBuilder parallelBuilder(sourceRegion.getContext());
        Block *parallelBlock = new Block();
        parallelBuilder.setInsertionPointToStart(parallelBlock);

        auto [allParallelized, copyprivateVars] =
            moveToSingle(std::get<SingleRegion>(opOrSingle), allocaBuilder,
                         singleBuilder, parallelBuilder);
        if (allParallelized) {
          // The single region was not required as all operations were safe to
          // parallelize
          assert(copyprivateVars.empty());
          assert(allocaBlock->empty());
          delete singleBlock;
        } else {
          omp::SingleOperands singleOperands;
          if (isLast)
            singleOperands.nowait = rootBuilder.getUnitAttr();
          singleOperands.copyprivateVars = copyprivateVars;
          cleanupBlock(singleBlock);
          for (auto var : singleOperands.copyprivateVars) {
            mlir::func::FuncOp funcOp =
                createCopyFunc(loc, var.getType(), firCopyFuncBuilder);
            singleOperands.copyprivateSyms.push_back(
                SymbolRefAttr::get(funcOp));
          }
          omp::SingleOp singleOp =
              rootBuilder.create<omp::SingleOp>(loc, singleOperands);
          singleOp.getRegion().push_back(singleBlock);
          targetRegion.front().getOperations().splice(
              singleOp->getIterator(), allocaBlock->getOperations());
        }
        rootBuilder.getInsertionBlock()->getOperations().splice(
            rootBuilder.getInsertionPoint(), parallelBlock->getOperations());
        delete allocaBlock;
        delete parallelBlock;
      } else {
        auto op = std::get<Operation *>(opOrSingle);
        if (auto wslw = dyn_cast<omp::WorkshareLoopWrapperOp>(op)) {
          omp::WsloopOperands wsloopOperands;
          if (isLast)
            wsloopOperands.nowait = rootBuilder.getUnitAttr();
          auto wsloop =
              rootBuilder.create<mlir::omp::WsloopOp>(loc, wsloopOperands);
          auto clonedWslw = cast<omp::WorkshareLoopWrapperOp>(
              rootBuilder.clone(*wslw, rootMapping));
          wsloop.getRegion().takeBody(clonedWslw.getRegion());
          clonedWslw->erase();
        } else {
          assert(mustParallelizeOp(op));
          Operation *cloned = rootBuilder.cloneWithoutRegions(*op, rootMapping);
          for (auto [region, clonedRegion] :
               llvm::zip(op->getRegions(), cloned->getRegions()))
            parallelizeRegion(region, clonedRegion, rootMapping, loc, di);
        }
      }
    }

    rootBuilder.clone(*block.getTerminator(), rootMapping);
  };

  if (sourceRegion.hasOneBlock()) {
    handleOneBlock(sourceRegion.front());
  } else if (!sourceRegion.empty()) {
    auto &domTree = di.getDomTree(&sourceRegion);
    for (auto node : llvm::breadth_first(domTree.getRootNode())) {
      handleOneBlock(*node->getBlock());
    }
  }

  for (Block &targetBlock : targetRegion)
    cleanupBlock(&targetBlock);
}

/// Lowers workshare to a sequence of single-thread regions and parallel loops
///
/// For example:
///
/// omp.workshare {
///   %a = fir.allocmem
///   omp.workshare.loop_wrapper {}
///   fir.call Assign %b %a
///   fir.freemem %a
/// }
///
/// becomes
///
/// %tmp = fir.alloca
/// omp.single copyprivate(%tmp) {
///   %a = fir.allocmem
///   fir.store %a %tmp
/// }
/// %a_reloaded = fir.load %tmp
/// omp.workshare.loop_wrapper {}
/// omp.single {
///   fir.call Assign %b %a_reloaded
///   fir.freemem %a_reloaded
/// }
///
/// Note that we allocate temporary memory for values in omp.single's which need
/// to be accessed by all threads and broadcast them using single's copyprivate
LogicalResult lowerWorkshare(mlir::omp::WorkshareOp wsOp, DominanceInfo &di) {
  Location loc = wsOp->getLoc();
  IRMapping rootMapping;

  OpBuilder rootBuilder(wsOp);

  // FIXME Currently, we only support workshare constructs with structured
  // control flow. The transformation itself supports CFG, however, once we
  // transform the MLIR region in the omp.workshare, we need to inline that
  // region in the parent block. We have no guarantees at this point of the
  // pipeline that the parent op supports CFG (e.g. fir.if), thus this is not
  // generally possible.  The alternative is to put the lowered region in an
  // operation akin to scf.execute_region, which will get lowered at the same
  // time when fir ops get lowered to CFG. However, SCF is not registered in
  // flang so we cannot use it. Remove this requirement once we have
  // scf.execute_region or an alternative operation available.
  if (wsOp.getRegion().getBlocks().size() == 1) {
    // This operation is just a placeholder which will be erased later. We need
    // it because our `parallelizeRegion` function works on regions and not
    // blocks.
    omp::WorkshareOp newOp =
        rootBuilder.create<omp::WorkshareOp>(loc, omp::WorkshareOperands());
    if (!wsOp.getNowait())
      rootBuilder.create<omp::BarrierOp>(loc);

    parallelizeRegion(wsOp.getRegion(), newOp.getRegion(), rootMapping, loc,
                      di);

    // Inline the contents of the placeholder workshare op into its parent
    // block.
    Block *theBlock = &newOp.getRegion().front();
    Operation *term = theBlock->getTerminator();
    Block *parentBlock = wsOp->getBlock();
    parentBlock->getOperations().splice(newOp->getIterator(),
                                        theBlock->getOperations());
    assert(term->getNumOperands() == 0);
    term->erase();
    newOp->erase();
    wsOp->erase();
  } else {
    // Otherwise just change the operation to an omp.single.

    wsOp->emitWarning(
        "omp workshare with unstructured control flow is currently "
        "unsupported and will be serialized.");

    // `shouldUseWorkshareLowering` should have guaranteed that there are no
    // omp.workshare_loop_wrapper's that bind to this omp.workshare.
    assert(!wsOp->walk([&](Operation *op) {
                  // Nested omp.workshare can have their own
                  // omp.workshare_loop_wrapper's.
                  if (isa<omp::WorkshareOp>(op))
                    return WalkResult::skip();
                  if (isa<omp::WorkshareLoopWrapperOp>(op))
                    return WalkResult::interrupt();
                  return WalkResult::advance();
                })
                .wasInterrupted());

    omp::SingleOperands operands;
    operands.nowait = wsOp.getNowaitAttr();
    omp::SingleOp newOp = rootBuilder.create<omp::SingleOp>(loc, operands);

    newOp.getRegion().getBlocks().splice(newOp.getRegion().getBlocks().begin(),
                                         wsOp.getRegion().getBlocks());
    wsOp->erase();
  }
  return success();
}

class LowerWorksharePass
    : public flangomp::impl::LowerWorkshareBase<LowerWorksharePass> {
public:
  void runOnOperation() override {
    mlir::DominanceInfo &di = getAnalysis<mlir::DominanceInfo>();
    getOperation()->walk([&](mlir::omp::WorkshareOp wsOp) {
      if (failed(lowerWorkshare(wsOp, di)))
        signalPassFailure();
    });
  }
};
} // namespace
