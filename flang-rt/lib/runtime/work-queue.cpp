//===-- lib/runtime/work-queue.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/work-queue.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Common/visit.h"

namespace Fortran::runtime {

#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
// FLANG_RT_DEBUG code is disabled when false.
static constexpr bool enableDebugOutput{false};
#endif

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS int Ticket::Continue(WorkQueue &workQueue) {
  if (!begun) {
    begun = true;
    return common::visit(
        [&workQueue](
            auto &specificTicket) { return specificTicket.Begin(workQueue); },
        u);
  } else {
    return common::visit(
        [&workQueue](auto &specificTicket) {
          return specificTicket.Continue(workQueue);
        },
        u);
  }
}

RT_API_ATTRS WorkQueue::~WorkQueue() {
  if (anyDynamicAllocation_) {
    if (last_) {
      if ((last_->next = firstFree_)) {
        last_->next->previous = last_;
      }
      firstFree_ = first_;
      first_ = last_ = nullptr;
    }
    while (firstFree_) {
      TicketList *next{firstFree_->next};
      if (!firstFree_->isStatic) {
        FreeMemory(firstFree_);
      }
      firstFree_ = next;
    }
  }
}

RT_API_ATTRS Ticket &WorkQueue::StartTicket() {
  if (!firstFree_) {
    void *p{AllocateMemoryOrCrash(terminator_, sizeof(TicketList))};
    firstFree_ = new (p) TicketList;
    firstFree_->isStatic = false;
    anyDynamicAllocation_ = true;
  }
  TicketList *newTicket{firstFree_};
  if ((firstFree_ = newTicket->next)) {
    firstFree_->previous = nullptr;
  }
  TicketList *after{insertAfter_ ? insertAfter_->next : nullptr};
  if ((newTicket->previous = insertAfter_ ? insertAfter_ : last_)) {
    newTicket->previous->next = newTicket;
  } else {
    first_ = newTicket;
  }
  if ((newTicket->next = after)) {
    after->previous = newTicket;
  } else {
    last_ = newTicket;
  }
  newTicket->ticket.begun = false;
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
  if (enableDebugOutput &&
      (executionEnvironment.internalDebugging &
          ExecutionEnvironment::WorkQueue)) {
    std::fprintf(stderr, "WQ: new ticket\n");
  }
#endif
  return newTicket->ticket;
}

RT_API_ATTRS int WorkQueue::Run() {
  while (last_) {
    TicketList *at{last_};
    insertAfter_ = last_;
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
    if (enableDebugOutput &&
        (executionEnvironment.internalDebugging &
            ExecutionEnvironment::WorkQueue)) {
      std::fprintf(stderr, "WQ: %zd %s\n", at->ticket.u.index(),
          at->ticket.begun ? "Continue" : "Begin");
    }
#endif
    int stat{at->ticket.Continue(*this)};
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
    if (enableDebugOutput &&
        (executionEnvironment.internalDebugging &
            ExecutionEnvironment::WorkQueue)) {
      std::fprintf(stderr, "WQ: ... stat %d\n", stat);
    }
#endif
    insertAfter_ = nullptr;
    if (stat == StatOk) {
      if (at->previous) {
        at->previous->next = at->next;
      } else {
        first_ = at->next;
      }
      if (at->next) {
        at->next->previous = at->previous;
      } else {
        last_ = at->previous;
      }
      if ((at->next = firstFree_)) {
        at->next->previous = at;
      }
      at->previous = nullptr;
      firstFree_ = at;
    } else if (stat != StatContinue) {
      Stop();
      return stat;
    }
  }
  return StatOk;
}

RT_API_ATTRS void WorkQueue::Stop() {
  if (last_) {
    if ((last_->next = firstFree_)) {
      last_->next->previous = last_;
    }
    firstFree_ = first_;
    first_ = last_ = nullptr;
  }
}

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
