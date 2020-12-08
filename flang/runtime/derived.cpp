//===-- runtime/derived.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "derived.h"
#include "descriptor.h"
#include "type-info.h"

namespace Fortran::runtime {

static const typeInfo::SpecialBinding *FindFinal(
    const typeInfo::DerivedType &derived, int rank) {
  const typeInfo::SpecialBinding *elemental{nullptr};
  const Descriptor &specialDesc{derived.special.descriptor()};
  std::size_t totalSpecialBindings{specialDesc.Elements()};
  for (std::size_t j{0}; j < totalSpecialBindings; ++j) {
    const auto &special{
        *specialDesc.ZeroBasedIndexedElement<typeInfo::SpecialBinding>(j)};
    switch (special.which) {
    case typeInfo::SpecialBinding::Which::Final:
      if (special.rank == rank) {
        return &special;
      }
      break;
    case typeInfo::SpecialBinding::Which::ElementalFinal:
      elemental = &special;
      break;
    case typeInfo::SpecialBinding::Which::AssumedRankFinal:
      return &special;
    default:;
    }
  }
  return elemental;
}

static void CallFinalSubroutine(
    const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
  if (const auto *special{FindFinal(derived, descriptor.rank())}) {
    if (special->which == typeInfo::SpecialBinding::Which::ElementalFinal) {
      std::size_t byteStride{descriptor.ElementBytes()};
      auto p{reinterpret_cast<void (*)(char *)>(special->proc)};
      // Finalizable objects must be contiguous.
      std::size_t elements{descriptor.Elements()};
      for (std::size_t j{0}; j < elements; ++j) {
        p(descriptor.OffsetElement<char>(j * byteStride));
      }
    } else if (special->isArgDescriptorSet & 1) {
      auto p{reinterpret_cast<void (*)(const Descriptor &)>(special->proc)};
      p(descriptor);
    } else {
      // Finalizable objects must be contiguous.
      auto p{reinterpret_cast<void (*)(char *)>(special->proc)};
      p(descriptor.OffsetElement<char>());
    }
  }
}

static inline SubscriptValue GetValue(
    const typeInfo::Value &value, const Descriptor &descriptor) {
  if (value.genre == typeInfo::Value::Genre::LenParameter) {
    return descriptor.Addendum()->LenParameterValue(value.value);
  } else {
    return value.value;
  }
}

// The order of finalization follows Fortran 2018 7.5.6.2, with
// deallocation of non-parent components (and their consequent finalization)
// taking place before parent component finalization.
void Destroy(const Descriptor &descriptor, bool finalize,
    const typeInfo::DerivedType &derived) {
  if (finalize) {
    CallFinalSubroutine(descriptor, derived);
  }
  const Descriptor &componentDesc{derived.component.descriptor()};
  std::int64_t myComponents{componentDesc.GetDimension(0).Extent()};
  std::size_t elements{descriptor.Elements()};
  std::size_t byteStride{descriptor.ElementBytes()};
  for (unsigned k{0}; k < myComponents; ++k) {
    const auto &comp{
        *componentDesc.ZeroBasedIndexedElement<typeInfo::Component>(k)};
    if (comp.genre == typeInfo::Component::Genre::Allocatable ||
        comp.genre == typeInfo::Component::Genre::Automatic) {
      for (std::size_t j{0}; j < elements; ++j) {
        descriptor.OffsetElement<Descriptor>(j * byteStride + comp.offset)
            ->Deallocate(finalize);
      }
    } else if (comp.genre == typeInfo::Component::Genre::Data &&
        comp.derivedType.descriptor().raw().base_addr) {
      SubscriptValue extent[maxRank];
      const Descriptor &boundsDesc{comp.bounds.descriptor()};
      for (int dim{0}; dim < comp.rank; ++dim) {
        extent[dim] =
            GetValue(
                *boundsDesc.ZeroBasedIndexedElement<typeInfo::Value>(2 * dim),
                descriptor) -
            GetValue(*boundsDesc.ZeroBasedIndexedElement<typeInfo::Value>(
                         2 * dim + 1),
                descriptor) +
            1;
      }
      StaticDescriptor<maxRank, true, 0> staticDescriptor;
      Descriptor &compDesc{staticDescriptor.descriptor()};
      const auto &compType{*comp.derivedType.descriptor()
                                .OffsetElement<typeInfo::DerivedType>()};
      for (std::size_t j{0}; j < elements; ++j) {
        compDesc.Establish(compType,
            descriptor.OffsetElement<char>(j * byteStride + comp.offset),
            comp.rank, extent);
        Destroy(compDesc, finalize, compType);
      }
    }
  }
  const Descriptor &parentDesc{derived.parent.descriptor()};
  if (const auto *parent{parentDesc.OffsetElement<typeInfo::DerivedType>()}) {
    Destroy(descriptor, finalize, *parent);
  }
}
} // namespace Fortran::runtime
