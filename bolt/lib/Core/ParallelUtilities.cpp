//===- bolt/Core/ParallelUtilities.cpp - Parallel utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the class that manages parallel work on BinaryFunctions.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Timer.h"
#include <mutex>

#define DEBUG_TYPE "par-utils"

namespace opts {
extern cl::OptionCategory BoltCategory;

cl::opt<unsigned>
ThreadCount("thread-count",
  cl::desc("number of threads"),
  cl::init(hardware_concurrency().compute_thread_count()),
  cl::cat(BoltCategory));

cl::opt<bool>
NoThreads("no-threads",
  cl::desc("disable multithreading"),
  cl::init(false),
  cl::cat(BoltCategory));

cl::opt<unsigned>
TaskCount("tasks-per-thread",
  cl::desc("number of tasks to be created per thread"),
  cl::init(20),
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {
namespace ParallelUtilities {

namespace {
/// A single thread pool that is used to run parallel tasks
std::unique_ptr<DefaultThreadPool> ThreadPoolPtr;

unsigned computeCostFor(const BinaryFunction &BF,
                        const PredicateTy &SkipPredicate,
                        const SchedulingPolicy &SchedPolicy) {
  if (SchedPolicy == SchedulingPolicy::SP_TRIVIAL)
    return 1;

  if (SkipPredicate && SkipPredicate(BF))
    return 0;

  switch (SchedPolicy) {
  case SchedulingPolicy::SP_CONSTANT:
    return 1;
  case SchedulingPolicy::SP_INST_LINEAR:
    return BF.getSize();
  case SchedulingPolicy::SP_INST_QUADRATIC:
    return BF.getSize() * BF.getSize();
  case SchedulingPolicy::SP_BB_LINEAR:
    return BF.size();
  case SchedulingPolicy::SP_BB_QUADRATIC:
    return BF.size() * BF.size();
  default:
    llvm_unreachable("unsupported scheduling policy");
  }
}

inline unsigned estimateTotalCost(const BinaryContext &BC,
                                  const PredicateTy &SkipPredicate,
                                  SchedulingPolicy &SchedPolicy) {
  if (SchedPolicy == SchedulingPolicy::SP_TRIVIAL)
    return BC.getBinaryFunctions().size();

  unsigned TotalCost = 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &BF = BFI.second;
    TotalCost += computeCostFor(BF, SkipPredicate, SchedPolicy);
  }

  // Switch to trivial scheduling if total estimated work is zero
  if (TotalCost == 0) {
    BC.outs()
        << "BOLT-WARNING: Running parallel work of 0 estimated cost, will "
           "switch to  trivial scheduling.\n";

    SchedPolicy = SP_TRIVIAL;
    TotalCost = BC.getBinaryFunctions().size();
  }
  return TotalCost;
}

} // namespace

ThreadPoolInterface &getThreadPool() {
  if (ThreadPoolPtr.get())
    return *ThreadPoolPtr;

  ThreadPoolPtr = std::make_unique<DefaultThreadPool>(
      llvm::hardware_concurrency(opts::ThreadCount));
  return *ThreadPoolPtr;
}

void runOnEachFunction(BinaryContext &BC, SchedulingPolicy SchedPolicy,
                       WorkFuncTy WorkFunction, PredicateTy SkipPredicate,
                       std::string LogName, bool ForceSequential,
                       unsigned TasksPerThread) {
  if (BC.getBinaryFunctions().size() == 0)
    return;

  auto runBlock = [&](std::map<uint64_t, BinaryFunction>::iterator BlockBegin,
                      std::map<uint64_t, BinaryFunction>::iterator BlockEnd) {
    Timer T(LogName, LogName);
    LLVM_DEBUG(T.startTimer());

    for (auto It = BlockBegin; It != BlockEnd; ++It) {
      BinaryFunction &BF = It->second;
      if (SkipPredicate && SkipPredicate(BF))
        continue;

      WorkFunction(BF);
    }
    LLVM_DEBUG(T.stopTimer());
  };

  if (opts::NoThreads || ForceSequential) {
    runBlock(BC.getBinaryFunctions().begin(), BC.getBinaryFunctions().end());
    return;
  }

  // Estimate the overall runtime cost using the scheduling policy
  const unsigned TotalCost = estimateTotalCost(BC, SkipPredicate, SchedPolicy);
  const unsigned BlocksCount = TasksPerThread * opts::ThreadCount;
  const unsigned BlockCost =
      TotalCost > BlocksCount ? TotalCost / BlocksCount : 1;

  // Divide work into blocks of equal cost
  ThreadPoolInterface &Pool = getThreadPool();
  auto BlockBegin = BC.getBinaryFunctions().begin();
  unsigned CurrentCost = 0;

  for (auto It = BC.getBinaryFunctions().begin();
       It != BC.getBinaryFunctions().end(); ++It) {
    BinaryFunction &BF = It->second;
    CurrentCost += computeCostFor(BF, SkipPredicate, SchedPolicy);

    if (CurrentCost >= BlockCost) {
      Pool.async(runBlock, BlockBegin, std::next(It));
      BlockBegin = std::next(It);
      CurrentCost = 0;
    }
  }
  Pool.async(runBlock, BlockBegin, BC.getBinaryFunctions().end());
  Pool.wait();
}

void runOnEachFunctionWithUniqueAllocId(
    BinaryContext &BC, SchedulingPolicy SchedPolicy,
    WorkFuncWithAllocTy WorkFunction, PredicateTy SkipPredicate,
    std::string LogName, bool ForceSequential, unsigned TasksPerThread) {
  if (BC.getBinaryFunctions().size() == 0)
    return;

  llvm::sys::RWMutex MainLock;
  auto runBlock = [&](std::map<uint64_t, BinaryFunction>::iterator BlockBegin,
                      std::map<uint64_t, BinaryFunction>::iterator BlockEnd,
                      MCPlusBuilder::AllocatorIdTy AllocId) {
    Timer T(LogName, LogName);
    LLVM_DEBUG(T.startTimer());
    std::shared_lock<llvm::sys::RWMutex> Lock(MainLock);
    for (auto It = BlockBegin; It != BlockEnd; ++It) {
      BinaryFunction &BF = It->second;
      if (SkipPredicate && SkipPredicate(BF))
        continue;

      WorkFunction(BF, AllocId);
    }
    LLVM_DEBUG(T.stopTimer());
  };

  unsigned AllocId = 1;
  auto EnsureAllocatorExists = [&BC](unsigned AllocId) {
    if (!BC.MIB->checkAllocatorExists(AllocId)) {
      MCPlusBuilder::AllocatorIdTy Id =
          BC.MIB->initializeNewAnnotationAllocator();
      (void)Id;
      assert(AllocId == Id && "unexpected allocator id created");
    }
  };

  if (opts::NoThreads || ForceSequential) {
    EnsureAllocatorExists(AllocId);
    runBlock(BC.getBinaryFunctions().begin(), BC.getBinaryFunctions().end(),
             AllocId);
    return;
  }
  // This lock is used to postpone task execution
  std::unique_lock<llvm::sys::RWMutex> Lock(MainLock);

  // Estimate the overall runtime cost using the scheduling policy
  const unsigned TotalCost = estimateTotalCost(BC, SkipPredicate, SchedPolicy);
  const unsigned BlocksCount = TasksPerThread * opts::ThreadCount;
  const unsigned BlockCost =
      TotalCost > BlocksCount ? TotalCost / BlocksCount : 1;

  // Divide work into blocks of equal cost
  ThreadPoolInterface &Pool = getThreadPool();
  auto BlockBegin = BC.getBinaryFunctions().begin();
  unsigned CurrentCost = 0;
  for (auto It = BC.getBinaryFunctions().begin();
       It != BC.getBinaryFunctions().end(); ++It) {
    BinaryFunction &BF = It->second;
    CurrentCost += computeCostFor(BF, SkipPredicate, SchedPolicy);

    if (CurrentCost >= BlockCost) {
      EnsureAllocatorExists(AllocId);
      Pool.async(runBlock, BlockBegin, std::next(It), AllocId);
      AllocId++;
      BlockBegin = std::next(It);
      CurrentCost = 0;
    }
  }

  EnsureAllocatorExists(AllocId);

  Pool.async(runBlock, BlockBegin, BC.getBinaryFunctions().end(), AllocId);
  Lock.unlock();
  Pool.wait();
}

} // namespace ParallelUtilities
} // namespace bolt
} // namespace llvm
