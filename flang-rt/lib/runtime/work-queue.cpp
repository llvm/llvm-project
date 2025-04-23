//===-- lib/runtime/work-queue.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/work-queue.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Common/visit.h"

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS ComponentTicketBase::ComponentTicketBase(
    const Descriptor &instance, const typeInfo::DerivedType &derived)
    : ElementalTicketBase{instance}, derived_{derived},
      components_{derived.component().Elements()} {}

RT_API_ATTRS bool ComponentTicketBase::CueUpNextItem() {
  bool elementsDone{!ElementalTicketBase::CueUpNextItem()};
  if (elementsDone) {
    component_ = nullptr;
    ++componentAt_;
  }
  if (!component_) {
    if (componentAt_ >= components_) {
      return false; // done!
    }
    const Descriptor &componentDesc{derived_.component()};
    component_ = componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(
        componentAt_);
    if (elementsDone) {
      ElementalTicketBase::Reset();
    }
  }
  return true;
}

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
      delete firstFree_;
    }
    firstFree_ = next;
  }
}

RT_API_ATTRS void WorkQueue::BeginInitialize(
    const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
  StartTicket().u.emplace<InitializeTicket>(descriptor, derived);
}

RT_API_ATTRS void WorkQueue::BeginInitializeClone(const Descriptor &clone,
    const Descriptor &original, const typeInfo::DerivedType &derived,
    bool hasStat, const Descriptor *errMsg) {
  StartTicket().u.emplace<InitializeCloneTicket>(
      clone, original, derived, hasStat, errMsg);
}

RT_API_ATTRS void WorkQueue::BeginFinalize(
    const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
  StartTicket().u.emplace<FinalizeTicket>(descriptor, derived);
}

RT_API_ATTRS void WorkQueue::BeginDestroy(const Descriptor &descriptor,
    const typeInfo::DerivedType &derived, bool finalize) {
  StartTicket().u.emplace<DestroyTicket>(descriptor, derived, finalize);
}

RT_API_ATTRS void WorkQueue::BeginAssign(
    Descriptor &to, const Descriptor &from, int flags, MemmoveFct memmoveFct) {
  StartTicket().u.emplace<AssignTicket>(to, from, flags, memmoveFct);
}

RT_API_ATTRS void WorkQueue::BeginDerivedAssign(Descriptor &to,
    const Descriptor &from, const typeInfo::DerivedType &derived, int flags,
    MemmoveFct memmoveFct, Descriptor *deallocateAfter) {
  StartTicket().u.emplace<DerivedAssignTicket>(
      to, from, derived, flags, memmoveFct, deallocateAfter);
}

RT_API_ATTRS Ticket &WorkQueue::StartTicket() {
  if (!firstFree_) {
    firstFree_ = new TicketList;
    firstFree_->isStatic = false;
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
  return newTicket->ticket;
}

RT_API_ATTRS int WorkQueue::Run() {
  while (last_) {
    TicketList *at{last_};
    insertAfter_ = last_;
    int stat{at->ticket.Continue(*this)};
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
    } else if (stat != StatOkContinue) {
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