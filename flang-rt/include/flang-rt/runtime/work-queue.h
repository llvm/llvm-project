//===-- include/flang-rt/runtime/work-queue.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Internal runtime utilities for work queues that replace the use of recursion
// for better GPU device support.
//
// A work queue is a list of tickets.  Each ticket class has a Begin()
// member function that is called once, and a Continue() member function
// that can be called zero or more times.  A ticket's execution terminates
// when either of these member functions returns a status other than
// StatOkContinue, and if that status is not StatOk, then the whole queue
// is shut down.
//
// By returning StatOkContinue from its Continue() member function,
// a ticket suspends its execution so that any nested tickets that it
// may have created can be run to completion.  It is the reponsibility
// of each ticket class to maintain resumption information in its state
// and manage its own progress.  Most ticket classes inherit from
// class ComponentTicketBase, which implements an outer loop over all
// components of a derived type, and an inner loop over all elements
// of a descriptor, possibly with multiple phases of execution per element.
//
// Tickets are created by WorkQueue::Begin...() member functions.
// There is one of these for each "top level" recursive function in the
// Fortran runtime support library that has been restructured into this
// ticket framework.
//
// When the work queue is running tickets, it always selects the last ticket
// on the list for execution -- "work stack" might have been a more accurate
// name for this framework.  This ticket may, while doing its job, create
// new tickets, and since those are pushed after the active one, the first
// such nested ticket will be the next one executed to completion -- i.e.,
// the order of nested WorkQueue::Begin...() calls is respected.
// Note that a ticket's Continue() member function won't be called again
// until all nested tickets have run to completion and it is once again
// the last ticket on the queue.
//
// Example for an assignment to a derived type:
// 1. Assign() is called, and its work queue is created.  It calls
//    WorkQueue::BeginAssign() and then WorkQueue::Run().
// 2. Run calls AssignTicket::Begin(), which pushes a tickets via
//    BeginFinalize() and returns StatOkContinue.
// 3. FinalizeTicket::Begin() and FinalizeTicket::Continue() are called
//    until one of them returns StatOk, which ends the finalization ticket.
// 4. AssignTicket::Continue() is then called; it creates a DerivedAssignTicket
//    and then returns StatOk, which ends the ticket.
// 5. At this point, only one ticket remains.  DerivedAssignTicket::Begin()
//    and ::Continue() are called until they are done (not StatOkContinue).
//    Along the way, it may create nested AssignTickets for components,
//    and suspend itself so that they may each run to completion.

#ifndef FLANG_RT_RUNTIME_WORK_QUEUE_H_
#define FLANG_RT_RUNTIME_WORK_QUEUE_H_

#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang/Common/api-attrs.h"
#include "flang/Runtime/freestanding-tools.h"
#include <flang/Common/variant.h>

namespace Fortran::runtime {
class Terminator;
class WorkQueue;
namespace typeInfo {
class DerivedType;
class Component;
} // namespace typeInfo

// Ticket workers

// Ticket workers return status codes.  Returning StatOkContinue means
// that the ticket is incomplete and must be resumed; any other value
// means that the ticket is complete, and if not StatOk, the whole
// queue can be shut down due to an error.
static constexpr int StatOkContinue{1234};

struct NullTicket {
  RT_API_ATTRS int Begin(WorkQueue &) const { return StatOk; }
  RT_API_ATTRS int Continue(WorkQueue &) const { return StatOk; }
};

// Base class for ticket workers that operate elementwise over descriptors
// TODO: if ComponentTicketBase remains this class' only client,
// merge them for better comprehensibility.
class ElementalTicketBase {
protected:
  RT_API_ATTRS ElementalTicketBase(const Descriptor &instance)
      : instance_{instance} {
    instance_.GetLowerBounds(subscripts_);
  }
  RT_API_ATTRS bool CueUpNextItem() const { return elementAt_ < elements_; }
  RT_API_ATTRS void AdvanceToNextElement() {
    phase_ = 0;
    ++elementAt_;
    instance_.IncrementSubscripts(subscripts_);
  }
  RT_API_ATTRS void Reset() {
    phase_ = 0;
    elementAt_ = 0;
    instance_.GetLowerBounds(subscripts_);
  }

  const Descriptor &instance_;
  std::size_t elements_{instance_.Elements()};
  std::size_t elementAt_{0};
  int phase_{0};
  SubscriptValue subscripts_[common::maxRank];
};

// Base class for ticket workers that operate over derived type components
// in an outer loop, and elements in an inner loop.
class ComponentTicketBase : protected ElementalTicketBase {
protected:
  RT_API_ATTRS ComponentTicketBase(
      const Descriptor &instance, const typeInfo::DerivedType &derived);
  RT_API_ATTRS bool CueUpNextItem();
  RT_API_ATTRS void AdvanceToNextComponent() { elementAt_ = elements_; }

  const typeInfo::DerivedType &derived_;
  const typeInfo::Component *component_{nullptr};
  std::size_t components_{0}, componentAt_{0};
  StaticDescriptor<common::maxRank, true, 0> componentDescriptor_;
};

// Implements derived type instance initialization
class InitializeTicket : private ComponentTicketBase {
public:
  RT_API_ATTRS InitializeTicket(
      const Descriptor &instance, const typeInfo::DerivedType &derived)
      : ComponentTicketBase{instance, derived} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);
};

// Initializes one derived type instance from the value of another
class InitializeCloneTicket : private ComponentTicketBase {
public:
  RT_API_ATTRS InitializeCloneTicket(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg)
      : ComponentTicketBase{original, derived}, clone_{clone},
        hasStat_{hasStat}, errMsg_{errMsg} {}
  RT_API_ATTRS int Begin(WorkQueue &) { return StatOkContinue; }
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const Descriptor &clone_;
  bool hasStat_{false};
  const Descriptor *errMsg_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> cloneComponentDescriptor_;
};

// Implements derived type instance finalization
class FinalizeTicket : private ComponentTicketBase {
public:
  RT_API_ATTRS FinalizeTicket(
      const Descriptor &instance, const typeInfo::DerivedType &derived)
      : ComponentTicketBase{instance, derived} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const typeInfo::DerivedType *finalizableParentType_{nullptr};
};

// Implements derived type instance destruction
class DestroyTicket : private ComponentTicketBase {
public:
  RT_API_ATTRS DestroyTicket(const Descriptor &instance,
      const typeInfo::DerivedType &derived, bool finalize)
      : ComponentTicketBase{instance, derived}, finalize_{finalize} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  bool finalize_{false};
};

// Implements general intrinsic assignment
class AssignTicket {
public:
  RT_API_ATTRS AssignTicket(
      Descriptor &to, const Descriptor &from, int flags, MemmoveFct memmoveFct)
      : to_{to}, from_{&from}, flags_{flags}, memmoveFct_{memmoveFct} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  RT_API_ATTRS bool IsSimpleMemmove() const {
    return !toDerived_ && to_.rank() == from_->rank() && to_.IsContiguous() &&
        from_->IsContiguous() && to_.ElementBytes() == from_->ElementBytes();
  }
  RT_API_ATTRS Descriptor &GetTempDescriptor();

  Descriptor &to_;
  const Descriptor *from_{nullptr};
  int flags_{0}; // enum AssignFlags
  MemmoveFct memmoveFct_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> tempDescriptor_;
  const typeInfo::DerivedType *toDerived_{nullptr};
  Descriptor *toDeallocate_{nullptr};
  bool persist_{false};
  bool done_{false};
};

// Implements derived type intrinsic assignment
class DerivedAssignTicket : private ComponentTicketBase {
public:
  RT_API_ATTRS DerivedAssignTicket(const Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter)
      : ComponentTicketBase{to, derived}, from_{from}, flags_{flags},
        memmoveFct_{memmoveFct}, deallocateAfter_{deallocateAfter} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);
  RT_API_ATTRS void AdvanceToNextElement();

private:
  const Descriptor &from_;
  int flags_{0};
  MemmoveFct memmoveFct_{nullptr};
  Descriptor *deallocateAfter_{nullptr};
  SubscriptValue fromSubscripts_[common::maxRank];
  StaticDescriptor<common::maxRank, true, 0> fromComponentDescriptor_;
};

struct Ticket {
  RT_API_ATTRS int Continue(WorkQueue &);
  bool begun{false};
  std::variant<NullTicket, InitializeTicket, InitializeCloneTicket,
      FinalizeTicket, DestroyTicket, AssignTicket, DerivedAssignTicket>
      u;
};

class WorkQueue {
public:
  RT_API_ATTRS explicit WorkQueue(Terminator &terminator)
      : terminator_{terminator} {
    for (int j{1}; j < numStatic_; ++j) {
      static_[j].previous = &static_[j - 1];
      static_[j - 1].next = &static_[j];
    }
  }
  RT_API_ATTRS ~WorkQueue();
  RT_API_ATTRS Terminator &terminator() { return terminator_; };

  RT_API_ATTRS void BeginInitialize(
      const Descriptor &descriptor, const typeInfo::DerivedType &derived);
  RT_API_ATTRS void BeginInitializeClone(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg);
  RT_API_ATTRS void BeginFinalize(
      const Descriptor &descriptor, const typeInfo::DerivedType &derived);
  RT_API_ATTRS void BeginDestroy(const Descriptor &descriptor,
      const typeInfo::DerivedType &derived, bool finalize);
  RT_API_ATTRS void BeginAssign(
      Descriptor &to, const Descriptor &from, int flags, MemmoveFct memmoveFct);
  RT_API_ATTRS void BeginDerivedAssign(Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter);

  RT_API_ATTRS int Run();

private:
  // Most uses of the work queue won't go very deep.
  static constexpr int numStatic_{2};

  struct TicketList {
    bool isStatic{true};
    Ticket ticket;
    TicketList *previous{nullptr}, *next{nullptr};
  };

  RT_API_ATTRS Ticket &StartTicket();
  RT_API_ATTRS void Stop();

  Terminator &terminator_;
  TicketList *first_{nullptr}, *last_{nullptr}, *insertAfter_{nullptr};
  TicketList static_[numStatic_];
  TicketList *firstFree_{static_};
};

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_WORK_QUEUE_H_
