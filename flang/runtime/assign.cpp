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
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

// Predicate: is the left-hand side of an assignment an allocated allocatable
// that must be deallocated?
static inline bool MustDeallocateLHS(
    Descriptor &to, const Descriptor &from, Terminator &terminator) {
  // Top-level assignments to allocatable variables (*not* components)
  // may first deallocate existing content if there's about to be a
  // change in type or shape; see F'2018 10.2.1.3(3).
  if (!to.IsAllocatable() || !to.IsAllocated()) {
    return false;
  }
  if (to.type() != from.type()) {
    return true;
  }
  DescriptorAddendum *toAddendum{to.Addendum()};
  const typeInfo::DerivedType *toDerived{
      toAddendum ? toAddendum->derivedType() : nullptr};
  const DescriptorAddendum *fromAddendum{from.Addendum()};
  const typeInfo::DerivedType *fromDerived{
      fromAddendum ? fromAddendum->derivedType() : nullptr};
  if (toDerived != fromDerived) {
    return true;
  }
  if (toAddendum) {
    // Distinct LEN parameters? Deallocate
    std::size_t lenParms{fromDerived ? fromDerived->LenParameters() : 0};
    for (std::size_t j{0}; j < lenParms; ++j) {
      if (toAddendum->LenParameterValue(j) !=
          fromAddendum->LenParameterValue(j)) {
        return true;
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
static int AllocateAssignmentLHS(
    Descriptor &to, const Descriptor &from, Terminator &terminator) {
  to.raw().type = from.raw().type;
  to.raw().elem_len = from.ElementBytes();
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
static void MaximalByteOffsetRange(
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
        least += extent * sm;
      } else {
        most += extent * sm;
      }
    }
  }
  most += desc.ElementBytes() - 1;
}

static inline bool RangesOverlap(const char *aStart, const char *aEnd,
    const char *bStart, const char *bEnd) {
  return aEnd >= bStart && bEnd >= aStart;
}

// Predicate: could the left-hand and right-hand sides of the assignment
// possibly overlap in memory?  Note that the descriptors themeselves
// are included in the test.
static bool MayAlias(const Descriptor &x, const Descriptor &y) {
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

static void DoScalarDefinedAssignment(const Descriptor &to,
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

static void DoElementalDefinedAssignment(const Descriptor &to,
    const Descriptor &from, const typeInfo::SpecialBinding &special) {
  SubscriptValue toAt[maxRank], fromAt[maxRank];
  to.GetLowerBounds(toAt);
  from.GetLowerBounds(fromAt);
  StaticDescriptor<maxRank, true, 8 /*?*/> statDesc[2];
  Descriptor &toElementDesc{statDesc[0].descriptor()};
  Descriptor &fromElementDesc{statDesc[1].descriptor()};
  toElementDesc = to;
  toElementDesc.raw().attribute = CFI_attribute_pointer;
  toElementDesc.raw().rank = 0;
  fromElementDesc = from;
  fromElementDesc.raw().attribute = CFI_attribute_pointer;
  fromElementDesc.raw().rank = 0;
  for (std::size_t toElements{to.Elements()}; toElements-- > 0;
       to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    toElementDesc.set_base_addr(to.Element<char>(toAt));
    fromElementDesc.set_base_addr(from.Element<char>(fromAt));
    DoScalarDefinedAssignment(toElementDesc, fromElementDesc, special);
  }
}

// Common implementation of assignments, both intrinsic assignments and
// those cases of polymorphic user-defined ASSIGNMENT(=) TBPs that could not
// be resolved in semantics.  Most assignment statements do not need any
// of the capabilities of this function -- but when the LHS is allocatable,
// the type might have a user-defined ASSIGNMENT(=), or the type might be
// finalizable, this function should be used.
static void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, bool maybeReallocate, bool needFinalization,
    bool canBeDefinedAssignment, bool componentCanBeDefinedAssignment) {
  bool mustDeallocateLHS{
      maybeReallocate && MustDeallocateLHS(to, from, terminator)};
  DescriptorAddendum *toAddendum{to.Addendum()};
  const typeInfo::DerivedType *toDerived{
      toAddendum ? toAddendum->derivedType() : nullptr};
  if (canBeDefinedAssignment && toDerived) {
    needFinalization &= !toDerived->noFinalizationNeeded();
    // Check for a user-defined assignment type-bound procedure;
    // see 10.2.1.4-5.  A user-defined assignment TBP defines all of
    // the semantics, including allocatable (re)allocation and any
    // finalization.
    if (to.rank() == 0) {
      if (const auto *special{toDerived->FindSpecialBinding(
              typeInfo::SpecialBinding::Which::ScalarAssignment)}) {
        return DoScalarDefinedAssignment(to, from, *special);
      }
    }
    if (const auto *special{toDerived->FindSpecialBinding(
            typeInfo::SpecialBinding::Which::ElementalAssignment)}) {
      return DoElementalDefinedAssignment(to, from, *special);
    }
  }
  bool isSimpleMemmove{!toDerived && to.rank() == from.rank() &&
      to.IsContiguous() && from.IsContiguous()};
  StaticDescriptor<maxRank, true, 10 /*?*/> deferredDeallocStatDesc;
  Descriptor *deferDeallocation{nullptr};
  if (MayAlias(to, from)) {
    if (mustDeallocateLHS) {
      deferDeallocation = &deferredDeallocStatDesc.descriptor();
      std::memcpy(deferDeallocation, &to, to.SizeInBytes());
      to.set_base_addr(nullptr);
    } else if (!isSimpleMemmove) {
      // Handle LHS/RHS aliasing by copying RHS into a temp, then
      // recursively assigning from that temp.
      auto descBytes{from.SizeInBytes()};
      StaticDescriptor<maxRank, true, 16> staticDesc;
      Descriptor &newFrom{staticDesc.descriptor()};
      std::memcpy(&newFrom, &from, descBytes);
      auto stat{ReturnError(terminator, newFrom.Allocate())};
      if (stat == StatOk) {
        char *toAt{newFrom.OffsetElement()};
        std::size_t fromElements{from.Elements()};
        std::size_t elementBytes{from.ElementBytes()};
        if (from.IsContiguous()) {
          std::memcpy(toAt, from.OffsetElement(), fromElements * elementBytes);
        } else {
          SubscriptValue fromAt[maxRank];
          for (from.GetLowerBounds(fromAt); fromElements-- > 0;
               toAt += elementBytes, from.IncrementSubscripts(fromAt)) {
            std::memcpy(toAt, from.Element<char>(fromAt), elementBytes);
          }
        }
        Assign(to, newFrom, terminator, /*maybeReallocate=*/false,
            needFinalization, false, componentCanBeDefinedAssignment);
        newFrom.Deallocate();
      }
      return;
    }
  }
  if (to.IsAllocatable()) {
    if (mustDeallocateLHS) {
      if (deferDeallocation) {
        if (needFinalization && toDerived) {
          Finalize(to, *toDerived);
          needFinalization = false;
        }
      } else {
        to.Destroy(/*finalize=*/needFinalization);
        needFinalization = false;
      }
    } else if (to.rank() != from.rank()) {
      terminator.Crash("Assign: mismatched ranks (%d != %d) in assignment to "
                       "unallocated allocatable",
          to.rank(), from.rank());
    }
    if (!to.IsAllocated()) {
      if (AllocateAssignmentLHS(to, from, terminator) != StatOk) {
        return;
      }
      needFinalization = false;
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
  std::size_t elementBytes{to.ElementBytes()};
  if (elementBytes != from.ElementBytes()) {
    terminator.Crash(
        "Assign: mismatching element sizes (to %zd bytes != from %zd bytes)",
        elementBytes, from.ElementBytes());
  }
  if (toDerived) {
    // Derived type intrinsic assignment, which is componentwise and elementwise
    // for all components, including parent components (10.2.1.2-3).
    // The target is first finalized if still necessary (7.5.6.3(1))
    if (needFinalization) {
      Finalize(to, *toDerived);
    }
    // Copy the data components (incl. the parent) first.
    const Descriptor &componentDesc{toDerived->component()};
    std::size_t numComponents{componentDesc.Elements()};
    for (std::size_t k{0}; k < numComponents; ++k) {
      const auto &comp{
          *componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(
              k)}; // TODO: exploit contiguity here
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
            Assign(toCompDesc, fromCompDesc, terminator,
                /*maybeReallocate=*/true,
                /*needFinalization=*/false, componentCanBeDefinedAssignment,
                componentCanBeDefinedAssignment);
          }
        } else { // Component has intrinsic type; simply copy raw bytes
          std::size_t componentByteSize{comp.SizeInBytes(to)};
          for (std::size_t j{0}; j < toElements; ++j,
               to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
            std::memmove(to.Element<char>(toAt) + comp.offset(),
                from.Element<const char>(fromAt) + comp.offset(),
                componentByteSize);
          }
        }
        break;
      case typeInfo::Component::Genre::Pointer: {
        std::size_t componentByteSize{comp.SizeInBytes(to)};
        for (std::size_t j{0}; j < toElements; ++j,
             to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
          std::memmove(to.Element<char>(toAt) + comp.offset(),
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
          if (toDesc->IsAllocatable()) {
            if (toDesc->IsAllocated()) {
              // Allocatable components of the LHS are unconditionally
              // deallocated before assignment (F'2018 10.2.1.3(13)(1)),
              // unlike a "top-level" assignment to a variable, where
              // deallocation is optional.
              // TODO: Consider skipping this step and deferring the
              // deallocation to the recursive activation of Assign(),
              // which might be able to avoid deallocation/reallocation
              // when the existing allocation can be reoccupied.
              toDesc->Destroy(false /*already finalized*/);
            }
            if (!fromDesc->IsAllocated()) {
              continue; // F'2018 10.2.1.3(13)(2)
            }
          }
          Assign(*toDesc, *fromDesc, terminator, /*maybeReallocate=*/true,
              /*needFinalization=*/false, componentCanBeDefinedAssignment,
              componentCanBeDefinedAssignment);
        }
        break;
      }
    }
    // Copy procedure pointer components
    const Descriptor &procPtrDesc{toDerived->procPtr()};
    std::size_t numProcPtrs{procPtrDesc.Elements()};
    for (std::size_t k{0}; k < numProcPtrs; ++k) {
      const auto &procPtr{
          *procPtrDesc.ZeroBasedIndexedElement<typeInfo::ProcPtrComponent>(k)};
      for (std::size_t j{0}; j < toElements; ++j, to.IncrementSubscripts(toAt),
           from.IncrementSubscripts(fromAt)) {
        std::memmove(to.Element<char>(toAt) + procPtr.offset,
            from.Element<const char>(fromAt) + procPtr.offset,
            sizeof(typeInfo::ProcedurePointer));
      }
    }
  } else { // intrinsic type, intrinsic assignment
    if (isSimpleMemmove) {
      std::memmove(
          to.raw().base_addr, from.raw().base_addr, toElements * elementBytes);
    } else { // elemental copies
      for (std::size_t n{toElements}; n-- > 0;
           to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
        std::memmove(to.Element<char>(toAt), from.Element<const char>(fromAt),
            elementBytes);
      }
    }
  }
  if (deferDeallocation) {
    deferDeallocation->Destroy();
  }
}

void DoFromSourceAssign(
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
        Assign(allocElement, source, terminator, /*maybeReallocate=*/false,
            /*needFinalization=*/false, false, false);
      }
    } else { // intrinsic type
      for (std::size_t n{alloc.Elements()}; n-- > 0;
           alloc.IncrementSubscripts(allocAt)) {
        std::memmove(alloc.Element<char>(allocAt), source.raw().base_addr,
            alloc.ElementBytes());
      }
    }
  } else {
    Assign(alloc, source, terminator, /*maybeReallocate=*/false,
        /*needFinalization=*/false, false, false);
  }
}

extern "C" {
void RTNAME(Assign)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  // All top-level defined assignments can be recognized in semantics and
  // will have been already been converted to calls, so don't check for
  // defined assignment apart from components.
  Assign(to, from, terminator, /*maybeReallocate=*/true,
      /*needFinalization=*/true,
      /*canBeDefinedAssignment=*/false,
      /*componentCanBeDefinedAssignment=*/true);
}

void RTNAME(AssignTemporary)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Assign(to, from, terminator, /*maybeReallocate=*/false,
      /*needFinalization=*/false,
      /*canBeDefinedAssignment=*/false,
      /*componentCanBeDefinedAssignment=*/false);
}

} // extern "C"
} // namespace Fortran::runtime
