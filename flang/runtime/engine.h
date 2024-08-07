//===-- runtime/engine.h ---------------------------------------*- C++ -*- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements a work engine for restartable tasks iterating over elements,
// components, &c. of arrays and derived types.  Avoids recursion and
// function pointers.

#ifndef FORTRAN_RUNTIME_ENGINE_H_
#define FORTRAN_RUNTIME_ENGINE_H_

#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime::engine {

class Engine;

// Every task object derives from Task.
struct Task {

  enum class ResultType { ResultValue /*doesn't matter*/ };

  Task(const Descriptor &instance, const typeInfo::DerivedType *derived)
      : instance_{instance}, derived_{derived} {}

  struct Iteration {
    RT_API_ATTRS bool Iterating(
        std::size_t iters, const Descriptor *dtor = nullptr) {
      if (!active) {
        if (iters > 0) {
          active = true;
          at = 0;
          n = iters;
          descriptor = dtor;
          if (descriptor) {
            descriptor->GetLowerBounds(subscripts);
          }
        }
      } else if (resuming) {
        resuming = false;
      } else if (++at < n) {
        if (descriptor) {
          descriptor->IncrementSubscripts(subscripts);
        }
      } else {
        active = false;
      }
      return active;
    }
    // Call on all Iteration instances before calling Engine::Begin()
    // when they should not advance when the job is resumed.
    RT_API_ATTRS void ResumeAtSameIteration() { resuming = true; }

    bool active{false}, resuming{false};
    std::size_t at, n;
    const Descriptor *descriptor;
    SubscriptValue subscripts[maxRank];
  };

  const Descriptor &instance_;
  const typeInfo::DerivedType *derived_;
  int phase_{0};

  // For looping over elements
  std::size_t elements_{instance_.Elements()};
  Iteration element_;

  // For looping over components
  const Descriptor *componentDesc_{derived_ ? &derived_->component() : nullptr};
  std::size_t components_{componentDesc_ ? componentDesc_->Elements() : 0};
  Iteration component_;
};

enum class Job { Initialization, Finalization, Destruction };

class Initialization : protected Task {
public:
  RT_API_ATTRS ResultType Resume(Engine &);

private:
  SubscriptValue extents_[maxRank];
  StaticDescriptor<maxRank, true, 0> staticDescriptor_;
};

class Finalization : protected Task {
public:
  RT_API_ATTRS ResultType Resume(Engine &);

private:
  SubscriptValue extents_[maxRank];
  StaticDescriptor<maxRank, true, 0> staticDescriptor_;
};

class Destruction : protected Task {
public:
  RT_API_ATTRS ResultType Resume(Engine &);

private:
  SubscriptValue extents_[maxRank];
  StaticDescriptor<maxRank, true, 0> staticDescriptor_;
};

class Engine {
public:
  RT_API_ATTRS Engine(
      Terminator &terminator, bool hasStat, const Descriptor *errMsg)
      : terminator_{terminator}, hasStat_{hasStat}, errMsg_{errMsg} {}
  RT_API_ATTRS ~Engine();

  RT_API_ATTRS Terminator &terminator() const { return terminator_; }
  RT_API_ATTRS bool hasStat() const { return hasStat_; }
  RT_API_ATTRS const Descriptor *errMsg() const { return errMsg_; }

  // Start and run a job to completion; returns status code.
  RT_API_ATTRS int Do(
      Job, const Descriptor &instance, const typeInfo::DerivedType *);

  // Callbacks from running tasks for use in their return statements.
  // Suspends execution and start a nested job
  RT_API_ATTRS Task::ResultType Begin(
      Job, const Descriptor &instance, const typeInfo::DerivedType *);
  // Terminates task successfully
  RT_API_ATTRS Task::ResultType Done();
  // Terminates task unsuccessfully
  RT_API_ATTRS Task::ResultType Fail(int status);

private:
  class Work {
  public:
    RT_API_ATTRS Work(
        Job job, const Descriptor &instance, const typeInfo::DerivedType *);
    RT_API_ATTRS void Resume(Engine &);

  private:
    Job job_;
    union {
      Task commonState;
      Initialization initialization;
      Finalization finalization;
      Destruction destruction;
    } u_;
  };

  struct WorkBlock {
    WorkBlock *previous{nullptr};
    OwningPtr<WorkBlock> next;
    int depth{0};
    static constexpr int maxDepth{4};
    alignas(Work) char workBuf[maxDepth][sizeof(Work)];
  };

  Terminator &terminator_;
  bool hasStat_{false};
  const Descriptor *errMsg_;
  int status_{StatOk};
  WorkBlock bottomWorkBlock_;
  WorkBlock *topWorkBlock_{&bottomWorkBlock_};
};

} // namespace Fortran::runtime::engine
#endif // FORTRAN_RUNTIME_ENGINE_H_
