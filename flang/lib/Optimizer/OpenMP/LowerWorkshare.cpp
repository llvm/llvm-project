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

#include <flang/Optimizer/Analysis/AliasAnalysis.h>
#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Dialect/FIROps.h>
#include <flang/Optimizer/Dialect/FIRType.h>
#include <flang/Optimizer/HLFIR/HLFIROps.h>
#include <flang/Optimizer/OpenMP/Passes.h>
#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
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

// If val is defined by an hlfir.declare/fir.declare, returns the declared
// memref (the value the declare wraps); otherwise returns val unchanged.
static Value lookThroughDeclare(Value val) {
  if (auto hlfirDecl = val.getDefiningOp<hlfir::DeclareOp>())
    return hlfirDecl.getMemref();
  if (auto firDecl = val.getDefiningOp<fir::DeclareOp>())
    return firDecl.getMemref();
  return val;
}

// Determines if a memory reference is thread-local in an OpenMP context.
//
// This is a best-effort analysis. We cannot definitively determine if code
// is inside a parallel region when it's in a function called from that
// region. However, we can identify common patterns of thread-local memory:
//
// 1. Memory allocated via fir.alloca inside the enclosing omp.parallel region
// 2. Memory that comes from OpenMP clause block arguments that create
//    thread-local storage (private, firstprivate, lastprivate, reduction,
//    linear clauses)
//
// Returns true if the memory reference appears to be thread-local and thus
// safe to parallelize (each thread should access its own copy).
static bool isOpenMPThreadLocalMemory(Operation *op, Value mem) {
  // Use AliasAnalysis to trace through declares, converts, reboxes, etc.
  // to find the underlying source of the memory reference.
  fir::AliasAnalysis aliasAnalysis;
  fir::AliasAnalysis::Source source = aliasAnalysis.getSource(mem);

  // Check if the source is a Value (not a global symbol).
  mlir::Value sourceValue =
      llvm::dyn_cast_if_present<mlir::Value>(source.origin.u);
  if (!sourceValue)
    return false;

  // Case 1: Memory allocated by fir.alloca inside the enclosing parallel
  // region is thread-private (each thread gets its own stack allocation).
  // Note: fir.allocmem is NOT thread-local even inside omp.parallel.
  if (source.kind == fir::AliasAnalysis::SourceKind::Allocate) {
    if (auto alloca = sourceValue.getDefiningOp<fir::AllocaOp>()) {
      if (auto parallelOp = alloca->getParentOfType<omp::ParallelOp>()) {
        if (op->getParentOfType<omp::ParallelOp>() == parallelOp)
          return true;
      }
    }
    // When the alias analysis encounters an hlfir.declare/fir.declare on a
    // private clause block argument, it marks the SourceKind as Allocate and
    // sets the source value to the declare op result (not the block arg).
    // Trace through the declare to check if the underlying Memref is a
    // private block argument.
    if (auto blockArg =
            llvm::dyn_cast<BlockArgument>(lookThroughDeclare(sourceValue))) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto argIface =
              llvm::dyn_cast<omp::BlockArgOpenMPOpInterface>(parentOp)) {
        if (llvm::is_contained(argIface.getPrivateBlockArgs(), blockArg))
          return true;
      }
    }
  }

  // Case 2: Memory from OpenMP clause block arguments that create thread-local
  // storage. These clauses create private copies for each thread:
  // - private: uninitialized thread-local copy
  // - firstprivate: thread-local copy initialized from original
  // - lastprivate: thread-local copy, value copied back after construct
  // - reduction: thread-local copy for reduction operations
  // - linear: thread-local copy with linear modification
  //
  // Check if the source value is a block argument of an OpenMP operation
  // that implements BlockArgOpenMPOpInterface.
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(sourceValue)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto argIface =
            llvm::dyn_cast<omp::BlockArgOpenMPOpInterface>(parentOp)) {
      // Check if this block argument corresponds to a privatizing clause.
      // Private, reduction, and in_reduction clauses create thread-local
      // memory.
      auto isInBlockArgs = [&](auto blockArgs) {
        return llvm::is_contained(blockArgs, blockArg);
      };

      if (isInBlockArgs(argIface.getPrivateBlockArgs()))
        return true;
      if (isInBlockArgs(argIface.getReductionBlockArgs()))
        return true;
      if (isInBlockArgs(argIface.getInReductionBlockArgs()))
        return true;
      if (isInBlockArgs(argIface.getTaskReductionBlockArgs()))
        return true;
    }
  }

  return false;
}

static bool isSafeToParallelize(Operation *op) {
  if (isa<hlfir::DeclareOp>(op) || isa<fir::DeclareOp>(op) ||
      isMemoryEffectFree(op))
    return true;

  // Thread-local variables allocated in the OpenMP parallel region or coming
  // from privatizing clauses are private to each thread and thus safe (and
  // sometimes required) to parallelize. If the compiler wraps stores to
  // thread-local variables in an omp.single block, only one thread updates
  // its local copy, while all other threads read uninitialized data (see
  // issue #143330).
  //
  // Only WRITE effects to thread-local memory are considered safe here, not
  // reads. If reads were also safe, the cascading effect in moveToSingle
  // could cause entire SingleRegions to become fully parallelized (all ops
  // safe), eliminating the omp.single and its implicit barrier. This removes
  // synchronization points needed to keep threads coordinated inside
  // sequential loops that contain workshared operations.
  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memEffects.getEffects(effects);
    if (!effects.empty() &&
        llvm::all_of(effects, [&](const MemoryEffects::EffectInstance &effect) {
          Value val = effect.getValue();
          return val && isa<MemoryEffects::Write>(effect.getEffect()) &&
                 isOpenMPThreadLocalMemory(op, val);
        }))
      return true;
  }

  return false;
}

// Returns the underlying thread-local storage that mem refers to, or null if
// mem is not thread-local. The alias analysis is used to look through
// fir.declare/hlfir.declare, fir.convert, fir.rebox, etc., so that two
// accesses of the same thread-local location yield the same value even if one
// goes through such ops and the other does not. This is what makes it safe to
// match the reads and writes of collect{Reads,Writes} against each other by
// value identity: a store to an alloca and a load from a fir.declare of that
// alloca map to the same key. Matching the raw effect value instead would
// silently miss such accesses, dropping a required broadcast.
static Value getOpenMPThreadLocalSource(Operation *op, Value mem) {
  if (!isOpenMPThreadLocalMemory(op, mem))
    return nullptr;
  fir::AliasAnalysis aliasAnalysis;
  Value source = llvm::dyn_cast_if_present<mlir::Value>(
      aliasAnalysis.getSource(mem).origin.u);
  if (!source)
    return nullptr;
  // The alias analysis looks through a fir.declare/hlfir.declare that wraps an
  // allocation, but stops at the declare result when it wraps a privatizing
  // clause block argument (see isOpenMPThreadLocalMemory). A write through such
  // a declare and a read of the block argument itself (or vice versa) would
  // then result in different origins and fail to match, dropping a required
  // broadcast. Look through the declare to the block argument so that both
  // accesses canonicalize to the same key.
  if (Value memref = lookThroughDeclare(source);
      llvm::isa<BlockArgument>(memref))
    return memref;
  return source;
}

// Collects the thread-local memory locations that op writes to and that
// need to be broadcasted to other threads when op ends up being executed
// by a single thread only.
//
// Some thread-local variables carry state which is logically shared by the
// whole omp.workshare region even though each thread owns a copy of it.
//
// One example is the fetch counter of the temporary storage used to implement
// FORALL: it is bumped from within an omp.single (because the value it is
// bumped by is only available there), so the copies owned by the threads
// which did not execute the omp.single would otherwise go stale and the
// following iterations would fetch the wrong element. See issue #209942.
//
// Only the underlying thread-local allocation is considered, so that a shallow
// copy of it faithfully reproduces the update on the other threads.
static void collectThreadLocalWrites(Operation *op,
                                     llvm::SmallVectorImpl<Value> &vars) {
  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects)
    return;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffects.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    Value val = effect.getValue();
    if (!val)
      continue;
    Value source = getOpenMPThreadLocalSource(op, val);
    if (!source)
      continue;
    auto refTy = dyn_cast<fir::ReferenceType>(source.getType());
    if (!refTy)
      continue;
    // createCopyFunc emits a load/store pair, so restrict this to types for
    // which such a shallow copy is both legal and cheap.
    mlir::Type eleTy = refTy.getEleTy();
    if (!fir::isa_trivial(eleTy) && !fir::isa_box_type(eleTy))
      continue;
    vars.push_back(source);
  }
}

// The thread-local locations that the whole team may read, used to decide
// which writes performed inside an omp.single must be broadcasted with
// copyprivate. "unknown" is set when an operation whose memory effects cannot
// be determined is found (see collectThreadLocalReads): such an operation
// might read any thread-local location, so every thread-local write then has
// to be broadcasted to stay correct.
struct ThreadLocalReads {
  llvm::SmallDenseSet<Value> reads;
  bool unknown = false;

  bool mayBeReadByTeam(Value v) const { return unknown || reads.contains(v); }
};

// Collects into reads the thread-local allocations that are read anywhere in
// scope. A thread-local location written from within an omp.single only needs
// to be broadcasted if some other thread may later read it. The scope must be
// a region executed by the whole team (i.e. the enclosing omp.parallel), so
// that reads performed after the omp.workshare region are accounted for too.
//
// Reads are matched by their underlying thread-local allocation, mirroring
// collectThreadLocalWrites, so that a load through a fir.declare/fir.convert
// still keeps the corresponding write live for broadcasting.
//
// An opaque call may read any thread-local location inside its callee, and a
// memory-effecting operation may report a read of unspecified memory (a read
// effect with no attached value). Neither read can be attributed to a specific
// location, so reads.unknown is set to force every thread-local write to be
// broadcasted. Other interface-less operations (e.g. omp.barrier, fir.declare)
// have known, inspectable behaviour and are safe to ignore here.
static void collectThreadLocalReads(Region &scope, ThreadLocalReads &reads) {
  scope.walk([&](Operation *op) {
    if (isa<mlir::CallOpInterface>(op)) {
      reads.unknown = true;
      return;
    }
    // TODO: An op that does not implement MemoryEffectOpInterface has
    // unknown effects and could read a thread-local location, so the
    // conservative choice would be to set reads.unknown here.
    // However, we deliberately don't, because the interface-less ops we see in
    // practice (omp.barrier, fir.declare, etc) have known, inspectable
    // behaviour and never read program memory. Forcing a broadcast for them
    // would defeat the read-back optimization. This assumption only holds for
    // the FIR/HLFIR/OpenMP dialects we know about; an op from another dialect
    // could break it. Once such ops are properly mapped (or made to model
    // their effects), fall back to reads.unknown here for any remaining
    // unknown-effect ops instead of ignoring them.
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffects)
      return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    memEffects.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Read>(effect.getEffect()))
        continue;
      Value val = effect.getValue();
      if (!val) {
        // A read effect without an attached value reads unspecified memory.
        reads.unknown = true;
        continue;
      }
      if (Value source = getOpenMPThreadLocalSource(op, val))
        reads.reads.insert(source);
    }
  });
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
      mlir::func::FuncOp::create(modBuilder, loc, copyFuncName, funcType);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  fir::factory::setInternalLinkage(funcOp);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      {loc, loc});
  builder.setInsertionPointToStart(&funcOp.getRegion().back());

  Value loaded = fir::LoadOp::create(builder, loc, funcOp.getArgument(1));
  fir::StoreOp::create(builder, loc, loaded, funcOp.getArgument(0));

  mlir::func::ReturnOp::create(builder, loc);
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

// canUseNowait to check whether the work generated for sourceRegion is the
// very last thing the omp.workshare region does, and thus whether the
// synchronization of its last omp.single/omp.wsloop may be left to the
// barrier emitted at the end of the omp.workshare region.
static void parallelizeRegion(Region &sourceRegion, Region &targetRegion,
                              IRMapping &rootMapping, Location loc,
                              mlir::DominanceInfo &di, bool canUseNowait,
                              const ThreadLocalReads &threadLocalReads) {
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
    Value alloc = fir::AllocaOp::create(allocaBuilder, loc, ty);
    fir::StoreOp::create(singleBuilder, loc, singleMapping.lookup(v), alloc);
    Value reloaded = fir::LoadOp::create(parallelBuilder, loc, ty, alloc);
    rootMapping.map(v, reloaded);
    return alloc;
  };

  auto moveToSingle =
      [&](SingleRegion sr, OpBuilder allocaBuilder, OpBuilder singleBuilder,
          OpBuilder parallelBuilder) -> std::pair<bool, SmallVector<Value>> {
    IRMapping singleMapping = rootMapping;
    SmallVector<Value> copyPrivate;
    // Thread-local memory updated by the single thread only, which has to be
    // broadcasted to the other threads to keep their copies in sync.
    SmallVector<Value> threadLocalWrites;
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
          // The operation only runs on the thread executing the omp.single,
          // so the thread-local memory it updates has to be broadcasted.
          collectThreadLocalWrites(&op, threadLocalWrites);
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
        collectThreadLocalWrites(&op, threadLocalWrites);
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
    omp::TerminatorOp::create(singleBuilder, loc);

    // Broadcast the thread-local state which only the thread executing the
    // omp.single has updated, but only when some other thread may actually read
    // it back: a location that is never read (e.g. a write to a temporary in a
    // terminal omp.single) does not need to be broadcasted. Values defined
    // inside sr are remapped; values defined before it (e.g. hoisted allocas)
    // are used as is.
    llvm::SmallDenseSet<Value> seen(copyPrivate.begin(), copyPrivate.end());
    for (Value v : threadLocalWrites) {
      if (!threadLocalReads.mayBeReadByTeam(v))
        continue;
      Value mapped = singleMapping.lookupOrDefault(v);
      if (seen.insert(mapped).second)
        copyPrivate.push_back(mapped);
    }

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

  auto handleOneBlock = [&](Block &block, bool blockCanUseNowait) {
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
      // Only the very last piece of work of the whole omp.workshare region
      // may use nowait and rely on the barrier emitted at the end of that
      // region.
      bool isLast = blockCanUseNowait && i + 1 == regions.size();
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
          // nowait and copyprivate are mutually exclusive on a single
          // construct: the broadcast relies on the barrier at the end of the
          // region.
          if (isLast && copyprivateVars.empty())
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
              omp::SingleOp::create(rootBuilder, loc, singleOperands);
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
              mlir::omp::WsloopOp::create(rootBuilder, loc, wsloopOperands);
          auto clonedWslw = cast<omp::WorkshareLoopWrapperOp>(
              rootBuilder.clone(*wslw, rootMapping));
          wsloop.getRegion().takeBody(clonedWslw.getRegion());
          clonedWslw->erase();
        } else {
          assert(mustParallelizeOp(op));
          // A loop-like operation may run its region more than once, so the
          // iterations of the work generated for it could overlap if nowait
          // were used inside of it.
          bool nestedCanUseNowait = isLast && !isa<LoopLikeOpInterface>(op);
          Operation *cloned = rootBuilder.cloneWithoutRegions(*op, rootMapping);
          for (auto [region, clonedRegion] :
               llvm::zip(op->getRegions(), cloned->getRegions()))
            parallelizeRegion(region, clonedRegion, rootMapping, loc, di,
                              nestedCanUseNowait, threadLocalReads);
        }
      }
    }

    rootBuilder.clone(*block.getTerminator(), rootMapping);
  };

  if (sourceRegion.hasOneBlock()) {
    handleOneBlock(sourceRegion.front(), canUseNowait);
  } else if (!sourceRegion.empty()) {
    // With several blocks, no block is known to hold the last piece of work of
    // the region, so none of them may use nowait.
    auto &domTree = di.getDomTree(&sourceRegion);
    for (auto node : llvm::breadth_first(domTree.getRootNode())) {
      handleOneBlock(*node->getBlock(), /*blockCanUseNowait=*/false);
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
        omp::WorkshareOp::create(rootBuilder, loc, omp::WorkshareOperands());
    if (!wsOp.getNowait())
      omp::BarrierOp::create(rootBuilder, loc);

    // Compute the thread-local locations read by the whole team, so that only
    // those get broadcasted out of the omp.single's below. The enclosing
    // omp.parallel is used as the scope so that reads performed after the
    // omp.workshare region are taken into account as well; if there is none,
    // fall back to the innermost isolated-from-above ancestor.
    ThreadLocalReads threadLocalReads;
    if (auto parallelOp = wsOp->getParentOfType<omp::ParallelOp>())
      collectThreadLocalReads(parallelOp.getRegion(), threadLocalReads);
    else if (Operation *top =
                 wsOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>())
      for (Region &r : top->getRegions())
        collectThreadLocalReads(r, threadLocalReads);

    parallelizeRegion(wsOp.getRegion(), newOp.getRegion(), rootMapping, loc, di,
                      /*canUseNowait=*/true, threadLocalReads);

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

    // If this was part of a combined construct (e.g. 'parallel workshare'), the
    // changes we just made to the region can be incompatible with a combined
    // construct, such as containing multiple block-associated constructs in it.
    if (auto parentOp =
            dyn_cast<omp::ComposableOpInterface>(parentBlock->getParentOp()))
      parentOp.setCombined(false);
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
    omp::SingleOp newOp = omp::SingleOp::create(rootBuilder, loc, operands);

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
