//===-- runtime/copy.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "copy.h"
#include "stack.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/descriptor.h"
#include <cstring>

namespace Fortran::runtime {
namespace {
using StaticDescTy = StaticDescriptor<maxRank, true, 0>;

// A structure describing the data copy that needs to be done
// from one descriptor to another. It is a helper structure
// for CopyElement.
struct CopyDescriptor {
  // A constructor specifying all members explicitly.
  RT_API_ATTRS CopyDescriptor(const Descriptor &to, const SubscriptValue toAt[],
      const Descriptor &from, const SubscriptValue fromAt[],
      std::size_t elements, bool usesStaticDescriptors = false)
      : to_(to), from_(from), elements_(elements),
        usesStaticDescriptors_(usesStaticDescriptors) {
    for (int dim{0}; dim < to.rank(); ++dim) {
      toAt_[dim] = toAt[dim];
    }
    for (int dim{0}; dim < from.rank(); ++dim) {
      fromAt_[dim] = fromAt[dim];
    }
  }
  // The number of elements to copy is initialized from the to descriptor.
  // The current element subscripts are initialized from the lower bounds
  // of the to and from descriptors.
  RT_API_ATTRS CopyDescriptor(const Descriptor &to, const Descriptor &from,
      bool usesStaticDescriptors = false)
      : to_(to), from_(from), elements_(to.Elements()),
        usesStaticDescriptors_(usesStaticDescriptors) {
    to.GetLowerBounds(toAt_);
    from.GetLowerBounds(fromAt_);
  }

  // Descriptor of the destination.
  const Descriptor &to_;
  // A subscript specifying the current element position to copy to.
  SubscriptValue toAt_[maxRank];
  // Descriptor of the source.
  const Descriptor &from_;
  // A subscript specifying the current element position to copy from.
  SubscriptValue fromAt_[maxRank];
  // Number of elements left to copy.
  std::size_t elements_;
  // Must be true, if the to and from descriptors are allocated
  // by the CopyElement runtime. The allocated memory belongs
  // to a separate stack that needs to be popped in correspondence
  // with popping such a CopyDescriptor node.
  bool usesStaticDescriptors_;
};

// A pair of StaticDescTy elements.
struct StaticDescriptorsPair {
  StaticDescTy to;
  StaticDescTy from;
};
} // namespace

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS void CopyElement(const Descriptor &to, const SubscriptValue toAt[],
    const Descriptor &from, const SubscriptValue fromAt[],
    Terminator &terminator) {
#if !defined(RT_DEVICE_COMPILATION)
  constexpr unsigned copyStackReserve{16};
  constexpr unsigned descriptorStackReserve{6};
#else
  // Always use dynamic allocation on the device to avoid
  // big stack sizes. This may be tuned as needed.
  constexpr unsigned copyStackReserve{0};
  constexpr unsigned descriptorStackReserve{0};
#endif
  // Keep a stack of CopyDescriptor's to avoid recursive calls.
  Stack<CopyDescriptor, copyStackReserve> copyStack{terminator};
  // Keep a separate stack of StaticDescTy pairs. These descriptors
  // may be used for representing copies of Component::Genre::Data
  // components (since they do not have their descriptors allocated
  // in memory).
  Stack<StaticDescriptorsPair, descriptorStackReserve> descriptorsStack{
      terminator};
  copyStack.emplace(to, toAt, from, fromAt, /*elements=*/std::size_t{1});

  while (!copyStack.empty()) {
    CopyDescriptor &currentCopy{copyStack.top()};
    std::size_t &elements{currentCopy.elements_};
    if (elements == 0) {
      // This copy has been exhausted.
      if (currentCopy.usesStaticDescriptors_) {
        // Pop the static descriptors, if they were used
        // for the current copy.
        descriptorsStack.pop();
      }
      copyStack.pop();
      continue;
    }
    const Descriptor &curTo{currentCopy.to_};
    SubscriptValue *curToAt{currentCopy.toAt_};
    const Descriptor &curFrom{currentCopy.from_};
    SubscriptValue *curFromAt{currentCopy.fromAt_};
    char *toPtr{curTo.Element<char>(curToAt)};
    char *fromPtr{curFrom.Element<char>(curFromAt)};
    RUNTIME_CHECK(terminator, curTo.ElementBytes() == curFrom.ElementBytes());
    // TODO: the memcpy can be optimized when both to and from are contiguous.
    // Moreover, if we came here from an Component::Genre::Data component,
    // all the per-element copies are redundant, because the parent
    // has already been copied as a whole.
    std::memcpy(toPtr, fromPtr, curTo.ElementBytes());
    --elements;
    if (elements != 0) {
      curTo.IncrementSubscripts(curToAt);
      curFrom.IncrementSubscripts(curFromAt);
    }

    // Deep copy allocatable and automatic components if any.
    if (const auto *addendum{curTo.Addendum()}) {
      if (const auto *derived{addendum->derivedType()};
          derived && !derived->noDestructionNeeded()) {
        RUNTIME_CHECK(terminator,
            curFrom.Addendum() && derived == curFrom.Addendum()->derivedType());
        const Descriptor &componentDesc{derived->component()};
        const typeInfo::Component *component{
            componentDesc.OffsetElement<typeInfo::Component>()};
        std::size_t nComponents{componentDesc.Elements()};
        for (std::size_t j{0}; j < nComponents; ++j, ++component) {
          if (component->genre() == typeInfo::Component::Genre::Allocatable ||
              component->genre() == typeInfo::Component::Genre::Automatic) {
            Descriptor &toDesc{
                *reinterpret_cast<Descriptor *>(toPtr + component->offset())};
            if (toDesc.raw().base_addr != nullptr) {
              toDesc.set_base_addr(nullptr);
              RUNTIME_CHECK(terminator, toDesc.Allocate() == CFI_SUCCESS);
              const Descriptor &fromDesc{*reinterpret_cast<const Descriptor *>(
                  fromPtr + component->offset())};
              copyStack.emplace(toDesc, fromDesc);
            }
          } else if (component->genre() == typeInfo::Component::Genre::Data &&
              component->derivedType() &&
              !component->derivedType()->noDestructionNeeded()) {
            SubscriptValue extents[maxRank];
            const typeInfo::Value *bounds{component->bounds()};
            std::size_t elements{1};
            for (int dim{0}; dim < component->rank(); ++dim) {
              SubscriptValue lb{bounds[2 * dim].GetValue(&curTo).value_or(0)};
              SubscriptValue ub{
                  bounds[2 * dim + 1].GetValue(&curTo).value_or(0)};
              extents[dim] = ub >= lb ? ub - lb + 1 : 0;
              elements *= extents[dim];
            }
            if (elements != 0) {
              const typeInfo::DerivedType &compType{*component->derivedType()};
              // Place a pair of static descriptors onto the descriptors stack.
              descriptorsStack.emplace();
              StaticDescriptorsPair &descs{descriptorsStack.top()};
              Descriptor &toCompDesc{descs.to.descriptor()};
              toCompDesc.Establish(compType, toPtr + component->offset(),
                  component->rank(), extents);
              Descriptor &fromCompDesc{descs.from.descriptor()};
              fromCompDesc.Establish(compType, fromPtr + component->offset(),
                  component->rank(), extents);
              copyStack.emplace(toCompDesc, fromCompDesc,
                  /*usesStaticDescriptors=*/true);
            }
          }
        }
      }
    }
  }
}
RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
