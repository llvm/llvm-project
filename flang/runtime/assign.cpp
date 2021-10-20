//===-- runtime/assign.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

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
    const Descriptor &from, const typeInfo::SpecialBinding &special,
    std::size_t toElements, SubscriptValue toAt[], SubscriptValue fromAt[]) {
  StaticDescriptor<maxRank, true, 8 /*?*/> statDesc[2];
  Descriptor &toElementDesc{statDesc[0].descriptor()};
  Descriptor &fromElementDesc{statDesc[1].descriptor()};
  toElementDesc = to;
  toElementDesc.raw().attribute = CFI_attribute_pointer;
  toElementDesc.raw().rank = 0;
  fromElementDesc = from;
  fromElementDesc.raw().attribute = CFI_attribute_pointer;
  fromElementDesc.raw().rank = 0;
  for (std::size_t j{0}; j < toElements;
       ++j, to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    toElementDesc.set_base_addr(to.Element<char>(toAt));
    fromElementDesc.set_base_addr(from.Element<char>(fromAt));
    DoScalarDefinedAssignment(toElementDesc, fromElementDesc, special);
  }
}

void Assign(Descriptor &to, const Descriptor &from, Terminator &terminator) {
  DescriptorAddendum *toAddendum{to.Addendum()};
  const typeInfo::DerivedType *toDerived{
      toAddendum ? toAddendum->derivedType() : nullptr};
  const DescriptorAddendum *fromAddendum{from.Addendum()};
  const typeInfo::DerivedType *fromDerived{
      fromAddendum ? fromAddendum->derivedType() : nullptr};
  bool wasJustAllocated{false};
  if (to.IsAllocatable()) {
    std::size_t lenParms{fromDerived ? fromDerived->LenParameters() : 0};
    if (to.IsAllocated()) {
      // Top-level assignments to allocatable variables (*not* components)
      // may first deallocate existing content if there's about to be a
      // change in type or shape; see F'2018 10.2.1.3(3).
      bool deallocate{false};
      if (to.type() != from.type()) {
        deallocate = true;
      } else if (toDerived != fromDerived) {
        deallocate = true;
      } else {
        if (toAddendum) {
          // Distinct LEN parameters? Deallocate
          for (std::size_t j{0}; j < lenParms; ++j) {
            if (toAddendum->LenParameterValue(j) !=
                fromAddendum->LenParameterValue(j)) {
              deallocate = true;
              break;
            }
          }
        }
        if (from.rank() > 0) {
          // Distinct shape? Deallocate
          int rank{to.rank()};
          for (int j{0}; j < rank; ++j) {
            if (to.GetDimension(j).Extent() != from.GetDimension(j).Extent()) {
              deallocate = true;
              break;
            }
          }
        }
      }
      if (deallocate) {
        to.Destroy(true /*finalize*/);
      }
    } else if (to.rank() != from.rank()) {
      terminator.Crash("Assign: mismatched ranks (%d != %d) in assignment to "
                       "unallocated allocatable",
          to.rank(), from.rank());
    }
    if (!to.IsAllocated()) {
      to.raw().type = from.raw().type;
      to.raw().elem_len = from.ElementBytes();
      if (toAddendum) {
        toDerived = fromDerived;
        toAddendum->set_derivedType(toDerived);
        for (std::size_t j{0}; j < lenParms; ++j) {
          toAddendum->SetLenParameterValue(
              j, fromAddendum->LenParameterValue(j));
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
      ReturnError(terminator, to.Allocate());
      if (fromDerived && !fromDerived->noInitializationNeeded()) {
        ReturnError(terminator, Initialize(to, *toDerived, terminator));
      }
      wasJustAllocated = true;
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
  if (toDerived) { // Derived type assignment
    // Check for defined assignment type-bound procedures (10.2.1.4-5)
    if (to.rank() == 0) {
      if (const auto *special{toDerived->FindSpecialBinding(
              typeInfo::SpecialBinding::Which::ScalarAssignment)}) {
        return DoScalarDefinedAssignment(to, from, *special);
      }
    }
    if (const auto *special{toDerived->FindSpecialBinding(
            typeInfo::SpecialBinding::Which::ElementalAssignment)}) {
      return DoElementalDefinedAssignment(
          to, from, *special, toElements, toAt, fromAt);
    }
    // Derived type intrinsic assignment, which is componentwise and elementwise
    // for all components, including parent components (10.2.1.2-3).
    // The target is first finalized if still necessary (7.5.6.3(1))
    if (!wasJustAllocated && !toDerived->noFinalizationNeeded()) {
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
            Assign(toCompDesc, fromCompDesc, terminator);
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
          Assign(*toDesc, *fromDesc, terminator);
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
    if (to.rank() == from.rank() && to.IsContiguous() && from.IsContiguous()) {
      // Everything is contiguous; do a single big copy
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
}

extern "C" {
void RTNAME(Assign)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Assign(to, from, terminator);
}

} // extern "C"
} // namespace Fortran::runtime
