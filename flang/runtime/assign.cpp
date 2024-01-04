//===-- runtime/assign.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "assign-impl.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

enum AssignFlags {
  NoAssignFlags = 0,
  MaybeReallocate = 1 << 0,
  NeedFinalization = 1 << 1,
  CanBeDefinedAssignment = 1 << 2,
  ComponentCanBeDefinedAssignment = 1 << 3,
  ExplicitLengthCharacterLHS = 1 << 4,
  PolymorphicLHS = 1 << 5,
  DeallocateLHS = 1 << 6
};

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
      if (to.GetDimension(j).Extent() != from.GetDimension(j).Extent()) {
        return true;
      }
    }
  }
  return false;
}

// Utility: allocate the allocatable left-hand side, either because it was
// originally deallocated or because it required reallocation
static RT_API_ATTRS int AllocateAssignmentLHS(
    Descriptor &to, const Descriptor &from, Terminator &terminator, int flags) {
  to.raw().type = from.raw().type;
  if (!(flags & ExplicitLengthCharacterLHS)) {
    to.raw().elem_len = from.ElementBytes();
  }
  const typeInfo::DerivedType *derived{nullptr};
  if (const DescriptorAddendum * fromAddendum{from.Addendum()}) {
    derived = fromAddendum->derivedType();
    if (DescriptorAddendum * toAddendum{to.Addendum()}) {
      toAddendum->set_derivedType(derived);
      std::size_t lenParms{derived ? derived->LenParameters() : 0};
      for (std::size_t j{0}; j < lenParms; ++j) {
        toAddendum->SetLenParameterValue(j, fromAddendum->LenParameterValue(j));
      }
    }
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
  int result{ReturnError(terminator, to.Allocate())};
  if (result == StatOk && derived && !derived->noInitializationNeeded()) {
    result = ReturnError(terminator, Initialize(to, *derived, terminator));
  }
  return result;
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
  const char *xDescLast{xDesc + x.SizeInBytes()};
  const char *yDesc{reinterpret_cast<const char *>(&y)};
  const char *yDescLast{yDesc + y.SizeInBytes()};
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
    const Descriptor &from, const typeInfo::SpecialBinding &special) {
  bool toIsDesc{special.IsArgDescriptor(0)};
  bool fromIsDesc{special.IsArgDescriptor(1)};
  if (toIsDesc) {
    if (fromIsDesc) {
      auto *p{
          special.GetProc<void (*)(const Descriptor &, const Descriptor &)>()};
      p(to, from);
    } else {
      auto *p{special.GetProc<void (*)(const Descriptor &, void *)>()};
      p(to, from.raw().base_addr);
    }
  } else {
    if (fromIsDesc) {
      auto *p{special.GetProc<void (*)(void *, const Descriptor &)>()};
      p(to.raw().base_addr, from);
    } else {
      auto *p{special.GetProc<void (*)(void *, void *)>()};
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
  for (std::size_t toElements{to.Elements()}; toElements-- > 0;
       to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    toElementDesc.set_base_addr(to.Element<char>(toAt));
    fromElementDesc.set_base_addr(from.Element<char>(fromAt));
    DoScalarDefinedAssignment(toElementDesc, fromElementDesc, special);
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
    Fortran::runtime::memmove(
        p, from.Element<std::add_const_t<CHAR>>(fromAt), fromElementBytes);
    p += copiedCharacters;
    for (auto n{padding}; n-- > 0;) {
      *p++ = CHAR{' '};
    }
  }
}

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
RT_API_ATTRS static void Assign(
    Descriptor &to, const Descriptor &from, Terminator &terminator, int flags) {
  bool mustDeallocateLHS{(flags & DeallocateLHS) ||
      MustDeallocateLHS(to, from, terminator, flags)};
  DescriptorAddendum *toAddendum{to.Addendum()};
  const typeInfo::DerivedType *toDerived{
      toAddendum ? toAddendum->derivedType() : nullptr};
  if (toDerived && (flags & NeedFinalization) &&
      toDerived->noFinalizationNeeded()) {
    flags &= ~NeedFinalization;
  }
  std::size_t toElementBytes{to.ElementBytes()};
  std::size_t fromElementBytes{from.ElementBytes()};
  // The following lambda definition violates the conding style,
  // but cuda-11.8 nvcc hits an internal error with the brace initialization.
  auto isSimpleMemmove = [&]() {
    return !toDerived && to.rank() == from.rank() && to.IsContiguous() &&
        from.IsContiguous() && toElementBytes == fromElementBytes;
  };
  StaticDescriptor<maxRank, true, 10 /*?*/> deferredDeallocStatDesc;
  Descriptor *deferDeallocation{nullptr};
  if (MayAlias(to, from)) {
    if (mustDeallocateLHS) {
      deferDeallocation = &deferredDeallocStatDesc.descriptor();
      std::memcpy(deferDeallocation, &to, to.SizeInBytes());
      to.set_base_addr(nullptr);
    } else if (!isSimpleMemmove()) {
      // Handle LHS/RHS aliasing by copying RHS into a temp, then
      // recursively assigning from that temp.
      auto descBytes{from.SizeInBytes()};
      StaticDescriptor<maxRank, true, 16> staticDesc;
      Descriptor &newFrom{staticDesc.descriptor()};
      std::memcpy(&newFrom, &from, descBytes);
      // Pretend the temporary descriptor is for an ALLOCATABLE
      // entity, otherwise, the Deallocate() below will not
      // free the descriptor memory.
      newFrom.raw().attribute = CFI_attribute_allocatable;
      auto stat{ReturnError(terminator, newFrom.Allocate())};
      if (stat == StatOk) {
        if (HasDynamicComponent(from)) {
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
          RTNAME(AssignTemporary)
          (newFrom, from, terminator.sourceFileName(), terminator.sourceLine());
        } else {
          ShallowCopy(newFrom, from, true, from.IsContiguous());
        }
        Assign(to, newFrom, terminator,
            flags &
                (NeedFinalization | ComponentCanBeDefinedAssignment |
                    ExplicitLengthCharacterLHS | CanBeDefinedAssignment));
        newFrom.Deallocate();
      }
      return;
    }
  }
  if (to.IsAllocatable()) {
    if (mustDeallocateLHS) {
      if (deferDeallocation) {
        if ((flags & NeedFinalization) && toDerived) {
          Finalize(to, *toDerived, &terminator);
          flags &= ~NeedFinalization;
        }
      } else {
        to.Destroy((flags & NeedFinalization) != 0, /*destroyPointers=*/false,
            &terminator);
        flags &= ~NeedFinalization;
      }
    } else if (to.rank() != from.rank() && !to.IsAllocated()) {
      terminator.Crash("Assign: mismatched ranks (%d != %d) in assignment to "
                       "unallocated allocatable",
          to.rank(), from.rank());
    }
    if (!to.IsAllocated()) {
      if (AllocateAssignmentLHS(to, from, terminator, flags) != StatOk) {
        return;
      }
      flags &= ~NeedFinalization;
      toElementBytes = to.ElementBytes(); // may have changed
    }
  }
  if (toDerived && (flags & CanBeDefinedAssignment)) {
    // Check for a user-defined assignment type-bound procedure;
    // see 10.2.1.4-5.  A user-defined assignment TBP defines all of
    // the semantics, including allocatable (re)allocation and any
    // finalization.
    //
    // Note that the aliasing and LHS (re)allocation handling above
    // needs to run even with CanBeDefinedAssignment flag, when
    // the Assign() is invoked recursively for component-per-component
    // assignments.
    if (to.rank() == 0) {
      if (const auto *special{toDerived->FindSpecialBinding(
              typeInfo::SpecialBinding::Which::ScalarAssignment)}) {
        return DoScalarDefinedAssignment(to, from, *special);
      }
    }
    if (const auto *special{toDerived->FindSpecialBinding(
            typeInfo::SpecialBinding::Which::ElementalAssignment)}) {
      return DoElementalDefinedAssignment(to, from, *toDerived, *special);
    }
  }
  SubscriptValue toAt[maxRank];
  to.GetLowerBounds(toAt);
  // Scalar expansion of the RHS is implied by using the same empty
  // subscript values on each (seemingly) elemental reference into
  // "from".
  SubscriptValue fromAt[maxRank];
  from.GetLowerBounds(fromAt);
  std::size_t toElements{to.Elements()};
  if (from.rank() > 0 && toElements != from.Elements()) {
    terminator.Crash("Assign: mismatching element counts in array assignment "
                     "(to %zd, from %zd)",
        toElements, from.Elements());
  }
  if (to.type() != from.type()) {
    terminator.Crash("Assign: mismatching types (to code %d != from code %d)",
        to.type().raw(), from.type().raw());
  }
  if (toElementBytes > fromElementBytes && !to.type().IsCharacter()) {
    terminator.Crash("Assign: mismatching non-character element sizes (to %zd "
                     "bytes != from %zd bytes)",
        toElementBytes, fromElementBytes);
  }
  if (const typeInfo::DerivedType *
      updatedToDerived{toAddendum ? toAddendum->derivedType() : nullptr}) {
    // Derived type intrinsic assignment, which is componentwise and elementwise
    // for all components, including parent components (10.2.1.2-3).
    // The target is first finalized if still necessary (7.5.6.3(1))
    if (flags & NeedFinalization) {
      Finalize(to, *updatedToDerived, &terminator);
    }
    // Copy the data components (incl. the parent) first.
    const Descriptor &componentDesc{updatedToDerived->component()};
    std::size_t numComponents{componentDesc.Elements()};
    for (std::size_t k{0}; k < numComponents; ++k) {
      const auto &comp{
          *componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(
              k)}; // TODO: exploit contiguity here
      // Use PolymorphicLHS for components so that the right things happen
      // when the components are polymorphic; when they're not, they're both
      // not, and their declared types will match.
      int nestedFlags{MaybeReallocate | PolymorphicLHS};
      if (flags & ComponentCanBeDefinedAssignment) {
        nestedFlags |= CanBeDefinedAssignment | ComponentCanBeDefinedAssignment;
      }
      switch (comp.genre()) {
      case typeInfo::Component::Genre::Data:
        if (comp.category() == TypeCategory::Derived) {
          StaticDescriptor<maxRank, true, 10 /*?*/> statDesc[2];
          Descriptor &toCompDesc{statDesc[0].descriptor()};
          Descriptor &fromCompDesc{statDesc[1].descriptor()};
          for (std::size_t j{0}; j < toElements; ++j,
               to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
            comp.CreatePointerDescriptor(toCompDesc, to, terminator, toAt);
            comp.CreatePointerDescriptor(
                fromCompDesc, from, terminator, fromAt);
            Assign(toCompDesc, fromCompDesc, terminator, nestedFlags);
          }
        } else { // Component has intrinsic type; simply copy raw bytes
          std::size_t componentByteSize{comp.SizeInBytes(to)};
          for (std::size_t j{0}; j < toElements; ++j,
               to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
            Fortran::runtime::memmove(to.Element<char>(toAt) + comp.offset(),
                from.Element<const char>(fromAt) + comp.offset(),
                componentByteSize);
          }
        }
        break;
      case typeInfo::Component::Genre::Pointer: {
        std::size_t componentByteSize{comp.SizeInBytes(to)};
        for (std::size_t j{0}; j < toElements; ++j,
             to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
          Fortran::runtime::memmove(to.Element<char>(toAt) + comp.offset(),
              from.Element<const char>(fromAt) + comp.offset(),
              componentByteSize);
        }
      } break;
      case typeInfo::Component::Genre::Allocatable:
      case typeInfo::Component::Genre::Automatic:
        for (std::size_t j{0}; j < toElements; ++j,
             to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
          auto *toDesc{reinterpret_cast<Descriptor *>(
              to.Element<char>(toAt) + comp.offset())};
          const auto *fromDesc{reinterpret_cast<const Descriptor *>(
              from.Element<char>(fromAt) + comp.offset())};
          // Allocatable components of the LHS are unconditionally
          // deallocated before assignment (F'2018 10.2.1.3(13)(1)),
          // unlike a "top-level" assignment to a variable, where
          // deallocation is optional.
          //
          // Be careful not to destroy/reallocate the LHS, if there is
          // overlap between LHS and RHS (it seems that partial overlap
          // is not possible, though).
          // Invoke Assign() recursively to deal with potential aliasing.
          if (toDesc->IsAllocatable()) {
            if (!fromDesc->IsAllocated()) {
              // No aliasing.
              //
              // If to is not allocated, the Destroy() call is a no-op.
              // This is just a shortcut, because the recursive Assign()
              // below would initiate the destruction for to.
              // No finalization is required.
              toDesc->Destroy(
                  /*finalize=*/false, /*destroyPointers=*/false, &terminator);
              continue; // F'2018 10.2.1.3(13)(2)
            }
          }
          // Force LHS deallocation with DeallocateLHS flag.
          // The actual deallocation may be avoided, if the existing
          // location can be reoccupied.
          Assign(*toDesc, *fromDesc, terminator, nestedFlags | DeallocateLHS);
        }
        break;
      }
    }
    // Copy procedure pointer components
    const Descriptor &procPtrDesc{updatedToDerived->procPtr()};
    std::size_t numProcPtrs{procPtrDesc.Elements()};
    for (std::size_t k{0}; k < numProcPtrs; ++k) {
      const auto &procPtr{
          *procPtrDesc.ZeroBasedIndexedElement<typeInfo::ProcPtrComponent>(k)};
      for (std::size_t j{0}; j < toElements; ++j, to.IncrementSubscripts(toAt),
           from.IncrementSubscripts(fromAt)) {
        Fortran::runtime::memmove(to.Element<char>(toAt) + procPtr.offset,
            from.Element<const char>(fromAt) + procPtr.offset,
            sizeof(typeInfo::ProcedurePointer));
      }
    }
  } else { // intrinsic type, intrinsic assignment
    if (isSimpleMemmove()) {
      Fortran::runtime::memmove(to.raw().base_addr, from.raw().base_addr,
          toElements * toElementBytes);
    } else if (toElementBytes > fromElementBytes) { // blank padding
      switch (to.type().raw()) {
      case CFI_type_signed_char:
      case CFI_type_char:
        BlankPadCharacterAssignment<char>(to, from, toAt, fromAt, toElements,
            toElementBytes, fromElementBytes);
        break;
      case CFI_type_char16_t:
        BlankPadCharacterAssignment<char16_t>(to, from, toAt, fromAt,
            toElements, toElementBytes, fromElementBytes);
        break;
      case CFI_type_char32_t:
        BlankPadCharacterAssignment<char32_t>(to, from, toAt, fromAt,
            toElements, toElementBytes, fromElementBytes);
        break;
      default:
        terminator.Crash("unexpected type code %d in blank padded Assign()",
            to.type().raw());
      }
    } else { // elemental copies, possibly with character truncation
      for (std::size_t n{toElements}; n-- > 0;
           to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
        Fortran::runtime::memmove(to.Element<char>(toAt),
            from.Element<const char>(fromAt), toElementBytes);
      }
    }
  }
  if (deferDeallocation) {
    // deferDeallocation is used only when LHS is an allocatable.
    // The finalization has already been run for it.
    deferDeallocation->Destroy(
        /*finalize=*/false, /*destroyPointers=*/false, &terminator);
  }
}

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS void DoFromSourceAssign(
    Descriptor &alloc, const Descriptor &source, Terminator &terminator) {
  if (alloc.rank() > 0 && source.rank() == 0) {
    // The value of each element of allocate object becomes the value of source.
    DescriptorAddendum *allocAddendum{alloc.Addendum()};
    const typeInfo::DerivedType *allocDerived{
        allocAddendum ? allocAddendum->derivedType() : nullptr};
    SubscriptValue allocAt[maxRank];
    alloc.GetLowerBounds(allocAt);
    if (allocDerived) {
      for (std::size_t n{alloc.Elements()}; n-- > 0;
           alloc.IncrementSubscripts(allocAt)) {
        Descriptor allocElement{*Descriptor::Create(*allocDerived,
            reinterpret_cast<void *>(alloc.Element<char>(allocAt)), 0)};
        Assign(allocElement, source, terminator, NoAssignFlags);
      }
    } else { // intrinsic type
      for (std::size_t n{alloc.Elements()}; n-- > 0;
           alloc.IncrementSubscripts(allocAt)) {
        Fortran::runtime::memmove(alloc.Element<char>(allocAt),
            source.raw().base_addr, alloc.ElementBytes());
      }
    }
  } else {
    Assign(alloc, source, terminator, NoAssignFlags);
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

  Assign(to, from, terminator, PolymorphicLHS);
}

void RTDEF(CopyOutAssign)(Descriptor &to, const Descriptor &from,
    bool skipToInit, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  // Initialize the "to" if it is of derived type that needs initialization.
  if (!skipToInit) {
    if (const DescriptorAddendum * addendum{to.Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noInitializationNeeded()) {
          if (ReturnError(terminator, Initialize(to, *derived, terminator)) !=
              StatOk) {
            return;
          }
        }
      }
    }
  }

  // Copyout from the temporary must not cause any finalizations
  // for LHS.
  Assign(to, from, terminator, NoAssignFlags);
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
