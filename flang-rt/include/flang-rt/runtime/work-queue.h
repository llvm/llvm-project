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
// A work queue comprises a list of tickets.  Each ticket class has a Begin()
// member function, which is called once, and a Continue() member function
// that can be called zero or more times.  A ticket's execution terminates
// when either of these member functions returns a status other than
// StatContinue.  When that status is not StatOk, then the whole queue
// is shut down.
//
// By returning StatContinue from its Continue() member function,
// a ticket suspends its execution so that any nested tickets that it
// may have created can be run to completion.  It is the reponsibility
// of each ticket class to maintain resumption information in its state
// and manage its own progress.  Most ticket classes inherit from
// class ComponentsOverElements, which implements an outer loop over all
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
//    BeginFinalize() and returns StatContinue.
// 3. FinalizeTicket::Begin() and FinalizeTicket::Continue() are called
//    until one of them returns StatOk, which ends the finalization ticket.
// 4. AssignTicket::Continue() is then called; it creates a DerivedAssignTicket
//    and then returns StatOk, which ends the ticket.
// 5. At this point, only one ticket remains.  DerivedAssignTicket::Begin()
//    and ::Continue() are called until they are done (not StatContinue).
//    Along the way, it may create nested AssignTickets for components,
//    and suspend itself so that they may each run to completion.

#ifndef FLANG_RT_RUNTIME_WORK_QUEUE_H_
#define FLANG_RT_RUNTIME_WORK_QUEUE_H_

#include "flang-rt/runtime/connection.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Common/api-attrs.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/freestanding-tools.h"
#include <flang/Common/variant.h>

namespace Fortran::runtime::io {
class IoStatementState;
struct NonTbpDefinedIoTable;
} // namespace Fortran::runtime::io

namespace Fortran::runtime {
class Terminator;
class WorkQueue;

RT_OFFLOAD_API_GROUP_BEGIN

// Ticket worker base classes

template <typename TICKET> class ImmediateTicketRunner {
public:
  RT_API_ATTRS explicit ImmediateTicketRunner(TICKET &ticket)
      : ticket_{ticket} {}
  RT_API_ATTRS int Run(WorkQueue &workQueue) {
    int status{ticket_.Begin(workQueue)};
    while (status == StatContinue) {
      status = ticket_.Continue(workQueue);
    }
    return status;
  }

private:
  TICKET &ticket_;
};

// Base class for ticket workers that operate elementwise over descriptors
class Elementwise {
public:
  RT_API_ATTRS Elementwise(
      const Descriptor &instance, const Descriptor *from = nullptr)
      : instance_{instance}, from_{from} {
    instance_.GetLowerBounds(subscripts_);
    if (from_) {
      from_->GetLowerBounds(fromSubscripts_);
    }
  }
  RT_API_ATTRS bool IsComplete() const { return elementAt_ >= elements_; }
  RT_API_ATTRS void Advance() {
    ++elementAt_;
    instance_.IncrementSubscripts(subscripts_);
    if (from_) {
      from_->IncrementSubscripts(fromSubscripts_);
    }
  }
  RT_API_ATTRS void SkipToEnd() { elementAt_ = elements_; }
  RT_API_ATTRS void Reset() {
    elementAt_ = 0;
    instance_.GetLowerBounds(subscripts_);
    if (from_) {
      from_->GetLowerBounds(fromSubscripts_);
    }
  }

protected:
  const Descriptor &instance_, *from_{nullptr};
  std::size_t elements_{instance_.InlineElements()};
  std::size_t elementAt_{0};
  SubscriptValue subscripts_[common::maxRank];
  SubscriptValue fromSubscripts_[common::maxRank];
};

// Base class for ticket workers that operate over derived type components.
class Componentwise {
public:
  RT_API_ATTRS Componentwise(const typeInfo::DerivedType &derived)
      : derived_{derived}, components_{derived_.component().InlineElements()} {
    GetFirstComponent();
  }

  RT_API_ATTRS bool IsComplete() const { return componentAt_ >= components_; }
  RT_API_ATTRS void Advance() {
    ++componentAt_;
    if (IsComplete()) {
      component_ = nullptr;
    } else {
      ++component_;
    }
  }
  RT_API_ATTRS void SkipToEnd() {
    component_ = nullptr;
    componentAt_ = components_;
  }
  RT_API_ATTRS void Reset() {
    component_ = nullptr;
    componentAt_ = 0;
    GetFirstComponent();
  }

protected:
  const typeInfo::DerivedType &derived_;
  std::size_t components_{0}, componentAt_{0};
  const typeInfo::Component *component_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> componentDescriptor_;

private:
  RT_API_ATTRS void GetFirstComponent() {
    if (components_ > 0) {
      component_ = derived_.component().OffsetElement<typeInfo::Component>();
    }
  }
};

// Base class for ticket workers that operate over derived type components
// in an outer loop, and elements in an inner loop.
class ComponentsOverElements : public Componentwise, public Elementwise {
public:
  RT_API_ATTRS ComponentsOverElements(const Descriptor &instance,
      const typeInfo::DerivedType &derived, const Descriptor *from = nullptr)
      : Componentwise{derived}, Elementwise{instance, from} {
    if (Elementwise::IsComplete()) {
      Componentwise::SkipToEnd();
    }
  }
  RT_API_ATTRS bool IsComplete() const { return Componentwise::IsComplete(); }
  RT_API_ATTRS void Advance() {
    SkipToNextElement();
    if (Elementwise::IsComplete()) {
      Elementwise::Reset();
      Componentwise::Advance();
    }
  }
  RT_API_ATTRS void SkipToNextElement() {
    phase_ = 0;
    Elementwise::Advance();
  }
  RT_API_ATTRS void SkipToNextComponent() {
    phase_ = 0;
    Elementwise::Reset();
    Componentwise::Advance();
  }
  RT_API_ATTRS void Reset() {
    phase_ = 0;
    Elementwise::Reset();
    Componentwise::Reset();
  }

protected:
  int phase_{0};
};

// Base class for ticket workers that operate over elements in an outer loop,
// type components in an inner loop.
class ElementsOverComponents : public Elementwise, public Componentwise {
public:
  RT_API_ATTRS ElementsOverComponents(const Descriptor &instance,
      const typeInfo::DerivedType &derived, const Descriptor *from = nullptr)
      : Elementwise{instance, from}, Componentwise{derived} {
    if (Componentwise::IsComplete()) {
      Elementwise::SkipToEnd();
    }
  }
  RT_API_ATTRS bool IsComplete() const { return Elementwise::IsComplete(); }
  RT_API_ATTRS void Advance() {
    SkipToNextComponent();
    if (Componentwise::IsComplete()) {
      Componentwise::Reset();
      Elementwise::Advance();
    }
  }
  RT_API_ATTRS void SkipToNextComponent() {
    phase_ = 0;
    Componentwise::Advance();
  }
  RT_API_ATTRS void SkipToNextElement() {
    phase_ = 0;
    Componentwise::Reset();
    Elementwise::Advance();
  }

protected:
  int phase_{0};
};

// Ticket worker classes

// Implements derived type instance initialization.
class InitializeTicket : public ImmediateTicketRunner<InitializeTicket>,
                         private ElementsOverComponents {
public:
  RT_API_ATTRS InitializeTicket(
      const Descriptor &instance, const typeInfo::DerivedType &derived)
      : ImmediateTicketRunner<InitializeTicket>{*this},
        ElementsOverComponents{instance, derived} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);
};

// Initializes one derived type instance from the value of another
class InitializeCloneTicket
    : public ImmediateTicketRunner<InitializeCloneTicket>,
      private ComponentsOverElements {
public:
  RT_API_ATTRS InitializeCloneTicket(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg)
      : ImmediateTicketRunner<InitializeCloneTicket>{*this},
        ComponentsOverElements{original, derived}, clone_{clone},
        hasStat_{hasStat}, errMsg_{errMsg} {}
  RT_API_ATTRS int Begin(WorkQueue &) { return StatContinue; }
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const Descriptor &clone_;
  bool hasStat_{false};
  const Descriptor *errMsg_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> cloneComponentDescriptor_;
};

// Implements derived type instance finalization
class FinalizeTicket : public ImmediateTicketRunner<FinalizeTicket>,
                       private ComponentsOverElements {
public:
  RT_API_ATTRS FinalizeTicket(
      const Descriptor &instance, const typeInfo::DerivedType &derived)
      : ImmediateTicketRunner<FinalizeTicket>{*this},
        ComponentsOverElements{instance, derived} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const typeInfo::DerivedType *finalizableParentType_{nullptr};
};

// Implements derived type instance destruction
class DestroyTicket : public ImmediateTicketRunner<DestroyTicket>,
                      private ComponentsOverElements {
public:
  RT_API_ATTRS DestroyTicket(const Descriptor &instance,
      const typeInfo::DerivedType &derived, bool finalize)
      : ImmediateTicketRunner<DestroyTicket>{*this},
        ComponentsOverElements{instance, derived}, finalize_{finalize},
        fixedStride_{instance.FixedStride()} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  bool finalize_{false};
  common::optional<SubscriptValue> fixedStride_;
};

// Implements general intrinsic assignment
class AssignTicket : public ImmediateTicketRunner<AssignTicket> {
public:
  RT_API_ATTRS AssignTicket(Descriptor &to, const Descriptor &from, int flags,
      MemmoveFct memmoveFct, const typeInfo::DerivedType *declaredType)
      : ImmediateTicketRunner<AssignTicket>{*this}, to_{to}, from_{&from},
        flags_{flags}, memmoveFct_{memmoveFct}, declaredType_{declaredType} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  RT_API_ATTRS Descriptor &GetTempDescriptor();
  RT_API_ATTRS bool IsSimpleMemmove() const {
    return !toDerived_ && to_.rank() == from_->rank() && to_.IsContiguous() &&
        from_->IsContiguous() && to_.ElementBytes() == from_->ElementBytes();
  }

  Descriptor &to_;
  const Descriptor *from_{nullptr};
  int flags_{0}; // enum AssignFlags
  MemmoveFct memmoveFct_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> tempDescriptor_;
  const typeInfo::DerivedType *declaredType_{nullptr};
  const typeInfo::DerivedType *toDerived_{nullptr};
  Descriptor *toDeallocate_{nullptr};
  bool persist_{false};
  bool done_{false};
};

// Implements derived type intrinsic assignment.
template <bool IS_COMPONENTWISE>
class DerivedAssignTicket
    : public ImmediateTicketRunner<DerivedAssignTicket<IS_COMPONENTWISE>>,
      private std::conditional_t<IS_COMPONENTWISE, ComponentsOverElements,
          ElementsOverComponents> {
public:
  using Base = std::conditional_t<IS_COMPONENTWISE, ComponentsOverElements,
      ElementsOverComponents>;
  RT_API_ATTRS DerivedAssignTicket(const Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter)
      : ImmediateTicketRunner<DerivedAssignTicket>{*this},
        Base{to, derived, &from}, flags_{flags}, memmoveFct_{memmoveFct},
        deallocateAfter_{deallocateAfter} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  static constexpr bool isComponentwise_{IS_COMPONENTWISE};
  bool toIsContiguous_{this->instance_.IsContiguous()};
  bool fromIsContiguous_{this->from_->IsContiguous()};
  int flags_{0};
  MemmoveFct memmoveFct_{nullptr};
  Descriptor *deallocateAfter_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> fromComponentDescriptor_;
};

namespace io::descr {

template <io::Direction DIR>
class DescriptorIoTicket
    : public ImmediateTicketRunner<DescriptorIoTicket<DIR>>,
      private Elementwise {
public:
  RT_API_ATTRS DescriptorIoTicket(io::IoStatementState &io,
      const Descriptor &descriptor, const io::NonTbpDefinedIoTable *table,
      bool &anyIoTookPlace)
      : ImmediateTicketRunner<DescriptorIoTicket>(*this),
        Elementwise{descriptor}, io_{io}, table_{table},
        anyIoTookPlace_{anyIoTookPlace} {}

  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);
  RT_API_ATTRS bool &anyIoTookPlace() { return anyIoTookPlace_; }

private:
  io::IoStatementState &io_;
  const io::NonTbpDefinedIoTable *table_{nullptr};
  bool &anyIoTookPlace_;
  common::optional<typeInfo::SpecialBinding> nonTbpSpecial_;
  const typeInfo::DerivedType *derived_{nullptr};
  const typeInfo::SpecialBinding *special_{nullptr};
  StaticDescriptor<common::maxRank, true, 0> elementDescriptor_;
};

template <io::Direction DIR>
class DerivedIoTicket : public ImmediateTicketRunner<DerivedIoTicket<DIR>>,
                        private ElementsOverComponents {
public:
  RT_API_ATTRS DerivedIoTicket(io::IoStatementState &io,
      const Descriptor &descriptor, const typeInfo::DerivedType &derived,
      const io::NonTbpDefinedIoTable *table, bool &anyIoTookPlace)
      : ImmediateTicketRunner<DerivedIoTicket>(*this),
        ElementsOverComponents{descriptor, derived}, io_{io}, table_{table},
        anyIoTookPlace_{anyIoTookPlace} {}
  RT_API_ATTRS int Begin(WorkQueue &) { return StatContinue; }
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  io::IoStatementState &io_;
  const io::NonTbpDefinedIoTable *table_{nullptr};
  bool &anyIoTookPlace_;
};

} // namespace io::descr

struct NullTicket {
  RT_API_ATTRS int Begin(WorkQueue &) const { return StatOk; }
  RT_API_ATTRS int Continue(WorkQueue &) const { return StatOk; }
};

struct Ticket {
  RT_API_ATTRS int Continue(WorkQueue &);
  bool begun{false};
  std::variant<NullTicket, InitializeTicket, InitializeCloneTicket,
      FinalizeTicket, DestroyTicket, AssignTicket, DerivedAssignTicket<false>,
      DerivedAssignTicket<true>,
      io::descr::DescriptorIoTicket<io::Direction::Output>,
      io::descr::DescriptorIoTicket<io::Direction::Input>,
      io::descr::DerivedIoTicket<io::Direction::Output>,
      io::descr::DerivedIoTicket<io::Direction::Input>>
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

  // APIs for particular tasks.  These can return StatOk if the work is
  // completed immediately.
  RT_API_ATTRS int BeginInitialize(
      const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
    if (runTicketsImmediately_) {
      return InitializeTicket{descriptor, derived}.Run(*this);
    } else {
      StartTicket().u.emplace<InitializeTicket>(descriptor, derived);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginInitializeClone(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg) {
    if (runTicketsImmediately_) {
      return InitializeCloneTicket{clone, original, derived, hasStat, errMsg}
          .Run(*this);
    } else {
      StartTicket().u.emplace<InitializeCloneTicket>(
          clone, original, derived, hasStat, errMsg);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginFinalize(
      const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
    if (runTicketsImmediately_) {
      return FinalizeTicket{descriptor, derived}.Run(*this);
    } else {
      StartTicket().u.emplace<FinalizeTicket>(descriptor, derived);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginDestroy(const Descriptor &descriptor,
      const typeInfo::DerivedType &derived, bool finalize) {
    if (runTicketsImmediately_) {
      return DestroyTicket{descriptor, derived, finalize}.Run(*this);
    } else {
      StartTicket().u.emplace<DestroyTicket>(descriptor, derived, finalize);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginAssign(Descriptor &to, const Descriptor &from,
      int flags, MemmoveFct memmoveFct,
      const typeInfo::DerivedType *declaredType) {
    if (runTicketsImmediately_) {
      return AssignTicket{to, from, flags, memmoveFct, declaredType}.Run(*this);
    } else {
      StartTicket().u.emplace<AssignTicket>(
          to, from, flags, memmoveFct, declaredType);
      return StatContinue;
    }
  }
  template <bool IS_COMPONENTWISE>
  RT_API_ATTRS int BeginDerivedAssign(Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter) {
    if (runTicketsImmediately_) {
      return DerivedAssignTicket<IS_COMPONENTWISE>{
          to, from, derived, flags, memmoveFct, deallocateAfter}
          .Run(*this);
    } else {
      StartTicket().u.emplace<DerivedAssignTicket<IS_COMPONENTWISE>>(
          to, from, derived, flags, memmoveFct, deallocateAfter);
      return StatContinue;
    }
  }
  template <io::Direction DIR>
  RT_API_ATTRS int BeginDescriptorIo(io::IoStatementState &io,
      const Descriptor &descriptor, const io::NonTbpDefinedIoTable *table,
      bool &anyIoTookPlace) {
    if (runTicketsImmediately_) {
      return io::descr::DescriptorIoTicket<DIR>{
          io, descriptor, table, anyIoTookPlace}
          .Run(*this);
    } else {
      StartTicket().u.emplace<io::descr::DescriptorIoTicket<DIR>>(
          io, descriptor, table, anyIoTookPlace);
      return StatContinue;
    }
  }
  template <io::Direction DIR>
  RT_API_ATTRS int BeginDerivedIo(io::IoStatementState &io,
      const Descriptor &descriptor, const typeInfo::DerivedType &derived,
      const io::NonTbpDefinedIoTable *table, bool &anyIoTookPlace) {
    if (runTicketsImmediately_) {
      return io::descr::DerivedIoTicket<DIR>{
          io, descriptor, derived, table, anyIoTookPlace}
          .Run(*this);
    } else {
      StartTicket().u.emplace<io::descr::DerivedIoTicket<DIR>>(
          io, descriptor, derived, table, anyIoTookPlace);
      return StatContinue;
    }
  }

  RT_API_ATTRS int Run();

private:
#if RT_DEVICE_COMPILATION
  // Always use the work queue on a GPU device to avoid recursion.
  static constexpr bool runTicketsImmediately_{false};
#else
  // Avoid the work queue overhead on the host, unless it needs
  // debugging, which is so much easier there.
  static constexpr bool runTicketsImmediately_{true};
#endif

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
  bool anyDynamicAllocation_{false};
};

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_WORK_QUEUE_H_
