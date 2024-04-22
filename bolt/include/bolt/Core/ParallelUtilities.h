//===- bolt/Core/ParallelUtilities.h - Parallel utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for assisting parallel processing of binary
// functions. Several scheduling criteria are supported using SchedulingPolicy,
// and are defined by how the runtime cost should be estimated. If the NoThreads
// flags is passed, all jobs will execute sequentially.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_PARALLEL_UTILITIES_H
#define BOLT_CORE_PARALLEL_UTILITIES_H

#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> ThreadCount;
extern cl::opt<bool> NoThreads;
extern cl::opt<unsigned> TaskCount;
} // namespace opts

namespace llvm {
namespace bolt {
class BinaryContext;
class BinaryFunction;

namespace ParallelUtilities {

using WorkFuncWithAllocTy =
    std::function<void(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy)>;
using WorkFuncTy = std::function<void(BinaryFunction &BF)>;
using PredicateTy = std::function<bool(const BinaryFunction &BF)>;

enum SchedulingPolicy {
  SP_TRIVIAL,     /// cost is estimated by the number of functions
  SP_CONSTANT,    /// cost is estimated by the number of non-skipped functions
  SP_INST_LINEAR, /// cost is estimated by inst count
  SP_INST_QUADRATIC, /// cost is estimated by the square of the inst count
  SP_BB_LINEAR,      /// cost is estimated by BB count
  SP_BB_QUADRATIC,   /// cost is estimated by the square of the BB count
};

/// Return the managed thread pool and initialize it if not initialized.
ThreadPoolInterface &getThreadPool();

/// Perform the work on each BinaryFunction except those that are accepted
/// by SkipPredicate, scheduling heuristic is based on SchedPolicy.
/// ForceSequential will selectively disable parallel execution and perform the
/// work sequentially.
void runOnEachFunction(BinaryContext &BC, SchedulingPolicy SchedPolicy,
                       WorkFuncTy WorkFunction,
                       PredicateTy SkipPredicate = PredicateTy(),
                       std::string LogName = "", bool ForceSequential = false,
                       unsigned TasksPerThread = opts::TaskCount);

/// Perform the work on each BinaryFunction except those that are rejected
/// by SkipPredicate, and create a unique annotation allocator for each
/// task. This should be used whenever the work function creates annotations to
/// allow thread-safe annotation creation.
/// ForceSequential will selectively disable parallel execution and perform the
/// work sequentially.
void runOnEachFunctionWithUniqueAllocId(
    BinaryContext &BC, SchedulingPolicy SchedPolicy,
    WorkFuncWithAllocTy WorkFunction, PredicateTy SkipPredicate,
    std::string LogName = "", bool ForceSequential = false,
    unsigned TasksPerThread = opts::TaskCount);

} // namespace ParallelUtilities
} // namespace bolt
} // namespace llvm
#endif
