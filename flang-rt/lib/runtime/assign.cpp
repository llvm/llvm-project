//===-- lib/runtime/assign.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"
#include "flang-rt/runtime/work-queue.h"

namespace Fortran::runtime {

// Predicate: is the left-hand side of an assignment an allocated allocatable
// that must be deallocated?
static inline RT_API_ATTRS bool MustDeallocateLHS(
    Descriptor &to, const Descriptor &from, Terminator &terminator, int flags) {
  // Top-level assignments to allocatable variables (*not* components)
  // may first deallocate existing content if there's about to be a
  // change in type or shape; see F'2018 10.2.1.3(3).
  if (!(flags & MaybeReallocate)) {
    return false;
  }
  if (!to.IsAllocatable() || !to.IsAllocated()) {
    return false;
  }
  if (to.type() != from.type()) {
    return true;
  }
  if (!(flags & ExplicitLengthCharacterLHS) && to.type().IsCharacter() &&
      to.ElementBytes() != from.ElementBytes()) {
    return true;
  }
  if (flags & PolymorphicLHS) {
    DescriptorAddendum *toAddendum{to.Addendum()};
    const typeInfo::DerivedType *toDerived{
        toAddendum ? toAddendum->derivedType() : nullptr};
    const DescriptorAddendum *fromAddendum{from.Addendum()};
    const typeInfo::DerivedType *fromDerived{
        fromAddendum ? fromAddendum->derivedType() : nullptr};
    if (toDerived != fromDerived) {
      return true;
    }
    if (fromDerived) {
      // Distinct LEN parameters? Deallocate
      std::size_t lenParms{fromDerived->LenParameters()};
      for (std::size_t j{0}; j < lenParms; ++j) {
        if (toAddendum->LenParameterValue(j) !=
            fromAddendum->LenParameterValue(j)) {
          return true;
        }
      }
    }
  }
  if (from.rank() > 0) {
    // Distinct shape? Deallocate
    int rank{to.rank()};
    for (int j{0}; j < rank; ++j) {
      const auto &toDim{to.GetDimension(j)};
      const auto &fromDim{from.GetDimension(j)};
      if (toDim.Extent() != fromDim.Extent()) {
        return true;
      }
      if ((flags & UpdateLHSBounds) &&
          toDim.LowerBound() != fromDim.LowerBound()) {
        return true;
      }
    }
  }
  // Not reallocating; may have to update bounds
  if (flags & UpdateLHSBounds) {
    int rank{to.rank()};
    for (int j{0}; j < rank; ++j) {
      to.GetDimension(j).SetLowerBound(from.GetDimension(j).LowerBound());
    }
  }
  return false;
}

// Utility: allocate the allocatable left-hand side, either because it was
// originally deallocated or because it required reallocation
static RT_API_ATTRS int AllocateAssignmentLHS(
    Descriptor &to, const Descriptor &from, Terminator &terminator, int flags) {
  DescriptorAddendum *toAddendum{to.Addendum()};
  const typeInfo::DerivedType *derived{nullptr};
  if (toAddendum) {
    derived = toAddendum->derivedType();
  }
  if (const DescriptorAddendum * fromAddendum{from.Addendum()}) {
    if (!derived || (flags & PolymorphicLHS)) {
      derived = fromAddendum->derivedType();
    }
    if (toAddendum && derived) {
      std::size_t lenParms{derived->LenParameters()};
      for (std::size_t j{0}; j < lenParms; ++j) {
        toAddendum->SetLenParameterValue(j, fromAddendum->LenParameterValue(j));
      }
    }
  } else {
    derived = nullptr;
  }
  if (toAddendum) {
    toAddendum->set_derivedType(derived);
  }
  to.raw().type = from.raw().type;
  if (derived) {
    to.raw().elem_len = derived->sizeInBytes();
  } else if (!(flags & ExplicitLengthCharacterLHS)) {
    to.raw().elem_len = from.ElementBytes();
  }
  // subtle: leave bounds in place when "from" is scalar (10.2.1.3(3))
  int rank{from.rank()};
  auto stride{static_cast<SubscriptValue>(to.ElementBytes())};
  for (int j{0}; j < rank; ++j) {
    auto &toDim{to.GetDimension(j)};
    const auto &fromDim{from.GetDimension(j)};
    toDim.SetBounds(fromDim.LowerBound(), fromDim.UpperBound());
    toDim.SetByteStride(stride);
    stride *= toDim.Extent();
  }
  return ReturnError(terminator, to.Allocate(kNoAsyncObject));
}

// least <= 0, most >= 0
static RT_API_ATTRS void MaximalByteOffsetRange(
    const Descriptor &desc, std::int64_t &least, std::int64_t &most) {
  least = most = 0;
  if (desc.ElementBytes() == 0) {
    return;
  }
  int n{desc.raw().rank};
  for (int j{0}; j < n; ++j) {
    const auto &dim{desc.GetDimension(j)};
    auto extent{dim.Extent()};
    if (extent > 0) {
      auto sm{dim.ByteStride()};
      if (sm < 0) {
        least += (extent - 1) * sm;
      } else {
        most += (extent - 1) * sm;
      }
    }
  }
  most += desc.ElementBytes() - 1;
}

static inline RT_API_ATTRS bool RangesOverlap(const char *aStart,
    const char *aEnd, const char *bStart, const char *bEnd) {
  return aEnd >= bStart && bEnd >= aStart;
}

// Predicate: could the left-hand and right-hand sides of the assignment
// possibly overlap in memory?  Note that the descriptors themeselves
// are included in the test.
static RT_API_ATTRS bool MayAlias(const Descriptor &x, const Descriptor &y) {
  const char *xBase{x.OffsetElement()};
  const char *yBase{y.OffsetElement()};
  if (!xBase || !yBase) {
    return false; // not both allocated
  }
  const char *xDesc{reinterpret_cast<const char *>(&x)};
  const char *xDescLast{xDesc + x.SizeInBytes() - 1};
  const char *yDesc{reinterpret_cast<const char *>(&y)};
  const char *yDescLast{yDesc + y.SizeInBytes() - 1};
  std::int64_t xLeast, xMost, yLeast, yMost;
  MaximalByteOffsetRange(x, xLeast, xMost);
  MaximalByteOffsetRange(y, yLeast, yMost);
  if (RangesOverlap(xDesc, xDescLast, yBase + yLeast, yBase + yMost) ||
      RangesOverlap(yDesc, yDescLast, xBase + xLeast, xBase + xMost)) {
    // A descriptor overlaps with the storage described by the other;
    // this can arise when an allocatable or pointer component is
    // being assigned to/from.
    return true;
  }
  if (!RangesOverlap(
          xBase + xLeast, xBase + xMost, yBase + yLeast, yBase + yMost)) {
    return false; // no storage overlap
  }
  // TODO: check dimensions: if any is independent, return false
  return true;
}

static RT_API_ATTRS void DoScalarDefinedAssignment(const Descriptor &to,
    const Descriptor &from, const typeInfo::DerivedType &derived,
    const typeInfo::SpecialBinding &special) {
  bool toIsDesc{special.IsArgDescriptor(0)};
  bool fromIsDesc{special.IsArgDescriptor(1)};
  const auto *bindings{
      derived.binding().OffsetElement<const typeInfo::Binding>()};
  if (toIsDesc) {
    if (fromIsDesc) {
      auto *p{special.GetProc<void (*)(const Descriptor &, const Descriptor &)>(
          bindings)};
      p(to, from);
    } else {
      auto *p{special.GetProc<void (*)(const Descriptor &, void *)>(bindings)};
      p(to, from.raw().base_addr);
    }
  } else {
    if (fromIsDesc) {
      auto *p{special.GetProc<void (*)(void *, const Descriptor &)>(bindings)};
      p(to.raw().base_addr, from);
    } else {
      auto *p{special.GetProc<void (*)(void *, void *)>(bindings)};
      p(to.raw().base_addr, from.raw().base_addr);
    }
  }
}

static RT_API_ATTRS void DoElementalDefinedAssignment(const Descriptor &to,
    const Descriptor &from, const typeInfo::DerivedType &derived,
    const typeInfo::SpecialBinding &special) {
  SubscriptValue toAt[maxRank], fromAt[maxRank];
  to.GetLowerBounds(toAt);
  from.GetLowerBounds(fromAt);
  StaticDescriptor<maxRank, true, 8 /*?*/> statDesc[2];
  Descriptor &toElementDesc{statDesc[0].descriptor()};
  Descriptor &fromElementDesc{statDesc[1].descriptor()};
  toElementDesc.Establish(derived, nullptr, 0, nullptr, CFI_attribute_pointer);
  fromElementDesc.Establish(
      derived, nullptr, 0, nullptr, CFI_attribute_pointer);
  for (std::size_t toElements{to.InlineElements()}; toElements-- > 0;
      to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    toElementDesc.set_base_addr(to.Element<char>(toAt));
    fromElementDesc.set_base_addr(from.Element<char>(fromAt));
    DoScalarDefinedAssignment(toElementDesc, fromElementDesc, derived, special);
  }
}

template <typename CHAR>
static RT_API_ATTRS void BlankPadCharacterAssignment(Descriptor &to,
    const Descriptor &from, SubscriptValue toAt[], SubscriptValue fromAt[],
    std::size_t elements, std::size_t toElementBytes,
    std::size_t fromElementBytes) {
  std::size_t padding{(toElementBytes - fromElementBytes) / sizeof(CHAR)};
  std::size_t copiedCharacters{fromElementBytes / sizeof(CHAR)};
  for (; elements-- > 0;
       to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    CHAR *p{to.Element<CHAR>(toAt)};
    runtime::memmove(
        p, from.Element<std::add_const_t<CHAR>>(fromAt), fromElementBytes);
    p += copiedCharacters;
    for (auto n{padding}; n-- > 0;) {
      *p++ = CHAR{' '};
    }
  }
}

RT_OFFLOAD_API_GROUP_BEGIN

// Common implementation of assignments, both intrinsic assignments and
// those cases of polymorphic user-defined ASSIGNMENT(=) TBPs that could not
// be resolved in semantics.  Most assignment statements do not need any
// of the capabilities of this function -- but when the LHS is allocatable,
// the type might have a user-defined ASSIGNMENT(=), or the type might be
// finalizable, this function should be used.
// When "to" is not a whole allocatable, "from" is an array, and defined
// assignments are not used, "to" and "from" only need to have the same number
// of elements, but their shape need not to conform (the assignment is done in
// element sequence order). This facilitates some internal usages, like when
// dealing with array constructors.
RT_API_ATTRS void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, MemmoveFct memmoveFct) {
  WorkQueue workQueue{terminator};
  if (workQueue.BeginAssign(to, from, flags, memmoveFct, nullptr) ==
      StatContinue) {
    workQueue.Run();
  }
}

RT_API_ATTRS int AssignTicket::Begin(WorkQueue &workQueue) {
  bool mustDeallocateLHS{(flags_ & DeallocateLHS) ||
      MustDeallocateLHS(to_, *from_, workQueue.terminator(), flags_)};
  DescriptorAddendum *toAddendum{to_.Addendum()};
  toDerived_ = toAddendum ? toAddendum->derivedType() : nullptr;
  if (toDerived_ && (flags_ & NeedFinalization) &&
      toDerived_->noFinalizationNeeded()) {
    flags_ &= ~NeedFinalization;
  }
  if (MayAlias(to_, *from_)) {
    if (mustDeallocateLHS) {
      // Convert the LHS into a temporary, then make it look deallocated.
      toDeallocate_ = &tempDescriptor_.descriptor();
      runtime::memcpy(
          reinterpret_cast<void *>(toDeallocate_), &to_, to_.SizeInBytes());
      to_.set_base_addr(nullptr);
      if (toDerived_ && (flags_ & NeedFinalization)) {
        int status{workQueue.BeginFinalize(*toDeallocate_, *toDerived_)};
        if (status == StatContinue) {
          // tempDescriptor_ state must outlive pending child ticket
          persist_ = true;
        } else if (status != StatOk) {
          return status;
        }
        flags_ &= ~NeedFinalization;
      }
    } else if (!IsSimpleMemmove()) {
      // Handle LHS/RHS aliasing by copying RHS into a temp, then
      // recursively assigning from that temp.
      auto descBytes{from_->SizeInBytes()};
      Descriptor &newFrom{tempDescriptor_.descriptor()};
      persist_ = true; // tempDescriptor_ state must outlive child tickets
      runtime::memcpy(reinterpret_cast<void *>(&newFrom), from_, descBytes);
      // Pretend the temporary descriptor is for an ALLOCATABLE
      // entity, otherwise, the Deallocate() below will not
      // free the descriptor memory.
      newFrom.raw().attribute = CFI_attribute_allocatable;
      if (int stat{ReturnError(
              workQueue.terminator(), newFrom.Allocate(kNoAsyncObject))};
          stat != StatOk) {
        if (stat == StatContinue) {
          persist_ = true;
        }
        return stat;
      }
      if (HasDynamicComponent(*from_)) {
        // If 'from' has allocatable/automatic component, we cannot
        // just make a shallow copy of the descriptor member.
        // This will still leave data overlap in 'to' and 'newFrom'.
        // For example:
        //   type t
        //     character, allocatable :: c(:)
        //   end type t
        //   type(t) :: x(3)
        //   x(2:3) = x(1:2)
        // We have to make a deep copy into 'newFrom' in this case.
        if (const DescriptorAddendum *addendum{newFrom.Addendum()}) {
          if (const auto *derived{addendum->derivedType()}) {
            if (!derived->noInitializationNeeded()) {
              if (int status{workQueue.BeginInitialize(newFrom, *derived)};
                  status != StatOk && status != StatContinue) {
                return status;
              }
            }
          }
        }
        static constexpr int nestedFlags{MaybeReallocate | PolymorphicLHS};
        if (int status{workQueue.BeginAssign(
                newFrom, *from_, nestedFlags, memmoveFct_, nullptr)};
            status != StatOk && status != StatContinue) {
          return status;
        }
      } else {
        ShallowCopy(newFrom, *from_, true, from_->IsContiguous());
      }
      from_ = &newFrom; // this is why from_ has to be a pointer
      flags_ &= NeedFinalization | ComponentCanBeDefinedAssignment |
          ExplicitLengthCharacterLHS | CanBeDefinedAssignment;
      toDeallocate_ = &newFrom;
    }
  }
  if (to_.IsAllocatable()) {
    if (mustDeallocateLHS) {
      if (!toDeallocate_ && to_.IsAllocated()) {
        toDeallocate_ = &to_;
      }
    } else if (to_.rank() != from_->rank() && !to_.IsAllocated()) {
      workQueue.terminator().Crash("Assign: mismatched ranks (%d != %d) in "
                                   "assignment to unallocated allocatable",
          to_.rank(), from_->rank());
    }
  } else if (!to_.IsAllocated()) {
    workQueue.terminator().Crash(
        "Assign: left-hand side variable is neither allocated nor allocatable");
  }
  if (toDerived_ && to_.IsAllocated()) {
    // Schedule finalization or destruction of the LHS.
    if (flags_ & NeedFinalization) {
      if (int status{workQueue.BeginFinalize(to_, *toDerived_)};
          status != StatOk && status != StatContinue) {
        return status;
      }
    } else if (!toDerived_->noDestructionNeeded()) {
      // F'2023 9.7.3.2 p7: "When an intrinsic assignment statement (10.2.1.3)
      // is executed, any noncoarray allocated allocatable subobject of the
      // variable is deallocated before the assignment takes place."
      if (int status{
              workQueue.BeginDestroy(to_, *toDerived_, /*finalize=*/false)};
          status != StatOk && status != StatContinue) {
        return status;
      }
    }
  }
  return StatContinue;
}

RT_API_ATTRS int AssignTicket::Continue(WorkQueue &workQueue) {
  if (done_) {
    // All child tickets are complete; can release this ticket's state.
    if (toDeallocate_) {
      toDeallocate_->Deallocate();
    }
    return StatOk;
  }
  // All necessary finalization or destruction that was initiated by Begin()
  // has been completed.  Deallocation may be pending, and if it's for the LHS,
  // do it now so that the LHS gets reallocated.
  if (toDeallocate_ == &to_) {
    toDeallocate_ = nullptr;
    to_.Deallocate();
  }
  // Allocate the LHS if needed
  if (!to_.IsAllocated()) {
    if (int stat{
            AllocateAssignmentLHS(to_, *from_, workQueue.terminator(), flags_)};
        stat != StatOk) {
      return stat;
    }
    const auto *addendum{to_.Addendum()};
    toDerived_ = addendum ? addendum->derivedType() : nullptr;
    if (toDerived_) {
      if (!toDerived_->noInitializationNeeded()) {
        if (int status{workQueue.BeginInitialize(to_, *toDerived_)};
            status != StatOk) {
          return status;
        }
      }
    }
  }
  // Check for a user-defined assignment type-bound procedure;
  // see 10.2.1.4-5.
  // Note that the aliasing and LHS (re)allocation handling above
  // needs to run even with CanBeDefinedAssignment flag, since
  // Assign() can be invoked recursively for component-wise assignments.
  // The declared type (if known) must be used for generic resolution
  // of ASSIGNMENT(=) to a binding, but that binding can be overridden.
  if (declaredType_ && (flags_ & CanBeDefinedAssignment)) {
    if (to_.rank() == 0) {
      if (const auto *special{declaredType_->FindSpecialBinding(
              typeInfo::SpecialBinding::Which::ScalarAssignment)}) {
        DoScalarDefinedAssignment(to_, *from_, *toDerived_, *special);
        done_ = true;
        return StatContinue;
      }
    }
    if (const auto *special{declaredType_->FindSpecialBinding(
            typeInfo::SpecialBinding::Which::ElementalAssignment)}) {
      DoElementalDefinedAssignment(to_, *from_, *toDerived_, *special);
      done_ = true;
      return StatContinue;
    }
  }
  // Intrinsic assignment
  std::size_t toElements{to_.InlineElements()};
  if (from_->rank() > 0) {
    std::size_t fromElements{from_->InlineElements()};
    if (toElements != fromElements) {
      workQueue.terminator().Crash("Assign: mismatching element counts in "
                                   "array assignment (to %zd, from %zd)",
          toElements, fromElements);
    }
  }
  if (to_.type() != from_->type()) {
    workQueue.terminator().Crash(
        "Assign: mismatching types (to code %d != from code %d)",
        to_.type().raw(), from_->type().raw());
  }
  std::size_t toElementBytes{to_.ElementBytes()};
  std::size_t fromElementBytes{from_->ElementBytes()};
  if (toElementBytes > fromElementBytes && !to_.type().IsCharacter()) {
    workQueue.terminator().Crash("Assign: mismatching non-character element "
                                 "sizes (to %zd bytes != from %zd bytes)",
        toElementBytes, fromElementBytes);
  }
  if (toDerived_) {
    if (toDerived_->noDefinedAssignment()) { // componentwise
      if (int status{workQueue.BeginDerivedAssign<true>(
              to_, *from_, *toDerived_, flags_, memmoveFct_, toDeallocate_)};
          status != StatOk && status != StatContinue) {
        return status;
      }
    } else { // elementwise
      if (int status{workQueue.BeginDerivedAssign<false>(
              to_, *from_, *toDerived_, flags_, memmoveFct_, toDeallocate_)};
          status != StatOk && status != StatContinue) {
        return status;
      }
    }
    toDeallocate_ = nullptr;
  } else if (IsSimpleMemmove()) {
    memmoveFct_(to_.raw().base_addr, from_->raw().base_addr,
        toElements * toElementBytes);
  } else {
    // Scalar expansion of the RHS is implied by using the same empty
    // subscript values on each (seemingly) elemental reference into
    // "from".
    SubscriptValue toAt[maxRank];
    to_.GetLowerBounds(toAt);
    SubscriptValue fromAt[maxRank];
    from_->GetLowerBounds(fromAt);
    if (toElementBytes > fromElementBytes) { // blank padding
      switch (to_.type().raw()) {
      case CFI_type_signed_char:
      case CFI_type_char:
        BlankPadCharacterAssignment<char>(to_, *from_, toAt, fromAt, toElements,
            toElementBytes, fromElementBytes);
        break;
      case CFI_type_char16_t:
        BlankPadCharacterAssignment<char16_t>(to_, *from_, toAt, fromAt,
            toElements, toElementBytes, fromElementBytes);
        break;
      case CFI_type_char32_t:
        BlankPadCharacterAssignment<char32_t>(to_, *from_, toAt, fromAt,
            toElements, toElementBytes, fromElementBytes);
        break;
      default:
        workQueue.terminator().Crash(
            "unexpected type code %d in blank padded Assign()",
            to_.type().raw());
      }
    } else { // elemental copies, possibly with character truncation
      for (std::size_t n{toElements}; n-- > 0;
          to_.IncrementSubscripts(toAt), from_->IncrementSubscripts(fromAt)) {
        memmoveFct_(to_.Element<char>(toAt), from_->Element<const char>(fromAt),
            toElementBytes);
      }
    }
  }
  if (persist_) {
    // tempDescriptor_ must outlive pending child ticket(s)
    done_ = true;
    return StatContinue;
  } else {
    if (toDeallocate_) {
      toDeallocate_->Deallocate();
      toDeallocate_ = nullptr;
    }
    return StatOk;
  }
}

template <bool IS_COMPONENTWISE>
RT_API_ATTRS int DerivedAssignTicket<IS_COMPONENTWISE>::Begin(
    WorkQueue &workQueue) {
  if (toIsContiguous_ && fromIsContiguous_ &&
      this->derived_.noDestructionNeeded() &&
      this->derived_.noDefinedAssignment() &&
      this->instance_.rank() == this->from_->rank()) {
    if (std::size_t elementBytes{this->instance_.ElementBytes()};
        elementBytes == this->from_->ElementBytes()) {
      // Fastest path.  Both LHS and RHS are contiguous, RHS is not a scalar
      // to be expanded, the types have the same size, and there are no
      // allocatable components or defined ASSIGNMENT(=) at any level.
      memmoveFct_(this->instance_.template OffsetElement<char>(),
          this->from_->template OffsetElement<const char *>(),
          this->instance_.InlineElements() * elementBytes);
      return StatOk;
    }
  }
  // Use PolymorphicLHS for components so that the right things happen
  // when the components are polymorphic; when they're not, they're both
  // not, and their declared types will match.
  int nestedFlags{MaybeReallocate | PolymorphicLHS};
  if (flags_ & ComponentCanBeDefinedAssignment) {
    nestedFlags |= CanBeDefinedAssignment | ComponentCanBeDefinedAssignment;
  }
  flags_ = nestedFlags;
  // Copy procedure pointer components
  const Descriptor &procPtrDesc{this->derived_.procPtr()};
  bool noDataComponents{this->IsComplete()};
  if (std::size_t numProcPtrs{procPtrDesc.InlineElements()}) {
    for (std::size_t k{0}; k < numProcPtrs; ++k) {
      const auto &procPtr{
          *procPtrDesc.ZeroBasedIndexedElement<typeInfo::ProcPtrComponent>(k)};
      // Loop only over elements
      if (k > 0) {
        Elementwise::Reset();
      }
      for (; !Elementwise::IsComplete(); Elementwise::Advance()) {
        memmoveFct_(this->instance_.template ElementComponent<char>(
                        this->subscripts_, procPtr.offset),
            this->from_->template ElementComponent<const char>(
                this->fromSubscripts_, procPtr.offset),
            sizeof(typeInfo::ProcedurePointer));
      }
    }
    if (noDataComponents) {
      return StatOk;
    }
    Elementwise::Reset();
  }
  if (noDataComponents) {
    return StatOk;
  }
  return StatContinue;
}
template RT_API_ATTRS int DerivedAssignTicket<false>::Begin(WorkQueue &);
template RT_API_ATTRS int DerivedAssignTicket<true>::Begin(WorkQueue &);

template <bool IS_COMPONENTWISE>
RT_API_ATTRS int DerivedAssignTicket<IS_COMPONENTWISE>::Continue(
    WorkQueue &workQueue) {
  while (!this->IsComplete()) {
    // Copy the data components (incl. the parent) first.
    switch (this->component_->genre()) {
    case typeInfo::Component::Genre::Data:
      if (this->component_->category() == TypeCategory::Derived) {
        Descriptor &toCompDesc{this->componentDescriptor_.descriptor()};
        Descriptor &fromCompDesc{this->fromComponentDescriptor_.descriptor()};
        this->component_->CreatePointerDescriptor(toCompDesc, this->instance_,
            workQueue.terminator(), this->subscripts_);
        this->component_->CreatePointerDescriptor(fromCompDesc, *this->from_,
            workQueue.terminator(), this->fromSubscripts_);
        const auto *componentDerived{this->component_->derivedType()};
        this->Advance();
        if (int status{workQueue.BeginAssign(toCompDesc, fromCompDesc, flags_,
                memmoveFct_, componentDerived)};
            status != StatOk) {
          return status;
        }
      } else { // Component has intrinsic type; simply copy raw bytes
        std::size_t componentByteSize{
            this->component_->SizeInBytes(this->instance_)};
        if (IS_COMPONENTWISE && toIsContiguous_ && fromIsContiguous_) {
          std::size_t offset{
              static_cast<std::size_t>(this->component_->offset())};
          char *to{this->instance_.template OffsetElement<char>(offset)};
          const char *from{
              this->from_->template OffsetElement<const char>(offset)};
          std::size_t toElementStride{this->instance_.ElementBytes()};
          std::size_t fromElementStride{
              this->from_->rank() == 0 ? 0 : this->from_->ElementBytes()};
          if (toElementStride == fromElementStride &&
              toElementStride == componentByteSize) {
            memmoveFct_(to, from, this->elements_ * componentByteSize);
          } else {
            for (std::size_t n{this->elements_}; n--;
                to += toElementStride, from += fromElementStride) {
              memmoveFct_(to, from, componentByteSize);
            }
          }
          this->SkipToNextComponent();
        } else {
          memmoveFct_(
              this->instance_.template Element<char>(this->subscripts_) +
                  this->component_->offset(),
              this->from_->template Element<const char>(this->fromSubscripts_) +
                  this->component_->offset(),
              componentByteSize);
          this->Advance();
        }
      }
      break;
    case typeInfo::Component::Genre::Pointer: {
      std::size_t componentByteSize{
          this->component_->SizeInBytes(this->instance_)};
      if (IS_COMPONENTWISE && toIsContiguous_ && fromIsContiguous_) {
        std::size_t offset{
            static_cast<std::size_t>(this->component_->offset())};
        char *to{this->instance_.template OffsetElement<char>(offset)};
        const char *from{
            this->from_->template OffsetElement<const char>(offset)};
        std::size_t toElementStride{this->instance_.ElementBytes()};
        std::size_t fromElementStride{
            this->from_->rank() == 0 ? 0 : this->from_->ElementBytes()};
        if (toElementStride == fromElementStride &&
            toElementStride == componentByteSize) {
          memmoveFct_(to, from, this->elements_ * componentByteSize);
        } else {
          for (std::size_t n{this->elements_}; n--;
              to += toElementStride, from += fromElementStride) {
            memmoveFct_(to, from, componentByteSize);
          }
        }
        this->SkipToNextComponent();
      } else {
        memmoveFct_(this->instance_.template Element<char>(this->subscripts_) +
                this->component_->offset(),
            this->from_->template Element<const char>(this->fromSubscripts_) +
                this->component_->offset(),
            componentByteSize);
        this->Advance();
      }
    } break;
    case typeInfo::Component::Genre::Allocatable:
    case typeInfo::Component::Genre::Automatic: {
      auto *toDesc{reinterpret_cast<Descriptor *>(
          this->instance_.template Element<char>(this->subscripts_) +
          this->component_->offset())};
      const auto *fromDesc{reinterpret_cast<const Descriptor *>(
          this->from_->template Element<char>(this->fromSubscripts_) +
          this->component_->offset())};
      const auto *componentDerived{this->component_->derivedType()};
      if (toDesc->IsAllocatable() && !fromDesc->IsAllocated()) {
        if (toDesc->IsAllocated()) {
          if (this->phase_ == 0) {
            if (componentDerived && !componentDerived->noDestructionNeeded()) {
              if (int status{workQueue.BeginDestroy(
                      *toDesc, *componentDerived, /*finalize=*/false)};
                  status != StatOk) {
                this->phase_++;
                return status;
              }
            }
          }
          toDesc->Deallocate();
        }
        this->Advance();
      } else {
        // Allocatable components of the LHS are unconditionally
        // deallocated before assignment (F'2018 10.2.1.3(13)(1)),
        // unlike a "top-level" assignment to a variable, where
        // deallocation is optional.
        int nestedFlags{flags_};
        if (!componentDerived ||
            (componentDerived->noFinalizationNeeded() &&
                componentDerived->noInitializationNeeded() &&
                componentDerived->noDestructionNeeded())) {
          // The actual deallocation might be avoidable when the existing
          // location can be reoccupied.
          nestedFlags |= MaybeReallocate | UpdateLHSBounds;
        } else {
          // Force LHS deallocation with DeallocateLHS flag.
          nestedFlags |= DeallocateLHS;
        }
        this->Advance();
        if (int status{workQueue.BeginAssign(*toDesc, *fromDesc, nestedFlags,
                memmoveFct_, componentDerived)};
            status != StatOk) {
          return status;
        }
      }
    } break;
    }
  }
  if (deallocateAfter_) {
    deallocateAfter_->Deallocate();
  }
  return StatOk;
}
template RT_API_ATTRS int DerivedAssignTicket<false>::Continue(WorkQueue &);
template RT_API_ATTRS int DerivedAssignTicket<true>::Continue(WorkQueue &);

RT_API_ATTRS void DoFromSourceAssign(Descriptor &alloc,
    const Descriptor &source, Terminator &terminator, MemmoveFct memmoveFct) {
  if (alloc.rank() > 0 && source.rank() == 0) {
    // The value of each element of allocate object becomes the value of source.
    DescriptorAddendum *allocAddendum{alloc.Addendum()};
    SubscriptValue allocAt[maxRank];
    alloc.GetLowerBounds(allocAt);
    std::size_t allocElementBytes{alloc.ElementBytes()};
    if (const typeInfo::DerivedType *allocDerived{
            allocAddendum ? allocAddendum->derivedType() : nullptr}) {
      // Handle derived type or short character source
      for (std::size_t n{alloc.InlineElements()}; n-- > 0;
          alloc.IncrementSubscripts(allocAt)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &allocElement{statDesc.descriptor()};
        allocElement.Establish(*allocDerived,
            reinterpret_cast<void *>(alloc.Element<char>(allocAt)), 0);
        Assign(allocElement, source, terminator, NoAssignFlags, memmoveFct);
      }
    } else if (allocElementBytes > source.ElementBytes()) {
      // Scalar expansion of short character source
      for (std::size_t n{alloc.InlineElements()}; n-- > 0;
          alloc.IncrementSubscripts(allocAt)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &allocElement{statDesc.descriptor()};
        allocElement.Establish(source.type(), allocElementBytes,
            reinterpret_cast<void *>(alloc.Element<char>(allocAt)), 0);
        Assign(allocElement, source, terminator, NoAssignFlags, memmoveFct);
      }
    } else { // intrinsic type scalar expansion, same data size
      for (std::size_t n{alloc.InlineElements()}; n-- > 0;
          alloc.IncrementSubscripts(allocAt)) {
        memmoveFct(alloc.Element<char>(allocAt), source.raw().base_addr,
            allocElementBytes);
      }
    }
  } else {
    Assign(alloc, source, terminator, NoAssignFlags, memmoveFct);
  }
}

RT_OFFLOAD_API_GROUP_END

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(Assign)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  // All top-level defined assignments can be recognized in semantics and
  // will have been already been converted to calls, so don't check for
  // defined assignment apart from components.
  Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment);
}

void RTDEF(AssignTemporary)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  // Initialize the "to" if it is of derived type that needs initialization.
  if (const DescriptorAddendum * addendum{to.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      // Do not invoke the initialization, if the descriptor is unallocated.
      // AssignTemporary() is used for component-by-component assignments,
      // for example, for structure constructors. This means that the LHS
      // may be an allocatable component with unallocated status.
      // The initialization will just fail in this case. By skipping
      // the initialization we let Assign() automatically allocate
      // and initialize the component according to the RHS.
      // So we only need to initialize the LHS here if it is allocated.
      // Note that initializing already initialized entity has no visible
      // effect, though, it is assumed that the compiler does not initialize
      // the temporary and leaves the initialization to this runtime code.
      if (!derived->noInitializationNeeded() && to.IsAllocated()) {
        if (ReturnError(terminator, Initialize(to, *derived, terminator)) !=
            StatOk) {
          return;
        }
      }
    }
  }
  Assign(to, from, terminator, MaybeReallocate | PolymorphicLHS);
}

void RTDEF(CopyInAssign)(Descriptor &temp, const Descriptor &var,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  temp = var;
  temp.set_base_addr(nullptr);
  temp.raw().attribute = CFI_attribute_allocatable;
  temp.Allocate(kNoAsyncObject);
  ShallowCopy(temp, var);
}

void RTDEF(CopyOutAssign)(
    Descriptor *var, Descriptor &temp, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  // Copyout from the temporary must not cause any finalizations
  // for LHS. The variable must be properly initialized already.
  if (var) {
    ShallowCopy(*var, temp);
  }
  temp.Deallocate();
}

void RTDEF(AssignExplicitLengthCharacter)(Descriptor &to,
    const Descriptor &from, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment |
          ExplicitLengthCharacterLHS);
}

void RTDEF(AssignPolymorphic)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment |
          PolymorphicLHS);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
