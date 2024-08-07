//===-- runtime/engine.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "engine.h"
#include "flang/Runtime/memory.h"

namespace Fortran::runtime::engine {

RT_API_ATTRS Engine::Work::Work(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived)
    : job_{job}, u_{{instance, derived}} {}

RT_API_ATTRS void Engine::Work::Resume(Engine &engine) {
  switch (job_) {
  case Job::Initialization:
    u_.initialization.Resume(engine);
    return;
  case Job::Finalization:
    u_.finalization.Resume(engine);
    return;
  case Job::Destruction:
    u_.destruction.Resume(engine);
    return;
  }
  engine.terminator().Crash(
      "Work::Run: bad job_ code %d", static_cast<int>(job_));
}

RT_API_ATTRS Engine::~Engine() {
  // deletes list owned by bottomWorkBlock_.next
}

RT_API_ATTRS int Engine::Do(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  Begin(job, instance, derived);
  while (topWorkBlock_ != &bottomWorkBlock_ || bottomWorkBlock_.depth > 0) {
    if (status_ == StatOk) {
      auto *w{reinterpret_cast<Work *>(
          topWorkBlock_->workBuf[topWorkBlock_->depth - 1])};
      w->Resume(*this);
    } else {
      Done();
    }
  }
  return status_;
}

RT_API_ATTRS Task::ResultType Engine::Begin(
    Job job, const Descriptor &instance, const typeInfo::DerivedType *derived) {
  if (topWorkBlock_->depth == topWorkBlock_->maxDepth) {
    if (!topWorkBlock_->next) {
      topWorkBlock_->next = New<WorkBlock>{terminator_}(topWorkBlock_);
    }
    topWorkBlock_ = topWorkBlock_->next.get();
  }
  new (topWorkBlock_->workBuf[topWorkBlock_->depth++])
      Work{job, instance, derived};
  return Task::ResultType::ResultValue;
}

RT_API_ATTRS Task::ResultType Engine::Done() {
  if (!--topWorkBlock_->depth) {
    if (auto *previous{topWorkBlock_->previous}) {
      topWorkBlock_ = previous;
    }
  }
  return Task::ResultType::ResultValue;
}

RT_API_ATTRS Task::ResultType Engine::Fail(int status) {
  status_ = status;
  return Done();
}

} // namespace Fortran::runtime::engine
