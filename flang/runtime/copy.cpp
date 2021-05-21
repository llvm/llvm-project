//===-- runtime/copy.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "copy.h"
#include "allocatable.h"
#include "descriptor.h"
#include "terminator.h"
#include "type-info.h"
#include <cstring>

namespace Fortran::runtime {

void CopyElement(const Descriptor &to, const SubscriptValue toAt[],
    const Descriptor &from, const SubscriptValue fromAt[],
    Terminator &terminator) {
  char *toPtr{to.Element<char>(toAt)};
  const char *fromPtr{from.Element<const char>(fromAt)};
  RUNTIME_CHECK(terminator, to.ElementBytes() == from.ElementBytes());
  std::memcpy(toPtr, fromPtr, to.ElementBytes());
  if (const auto *addendum{to.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      RUNTIME_CHECK(terminator,
          from.Addendum() && derived == from.Addendum()->derivedType());
      const Descriptor &componentDesc{derived->component.descriptor()};
      const typeInfo::Component *component{
          componentDesc.OffsetElement<typeInfo::Component>()};
      std::size_t nComponents{componentDesc.Elements()};
      for (std::size_t j{0}; j < nComponents; ++j, ++component) {
        if (component->genre == typeInfo::Component::Genre::Allocatable ||
            component->genre == typeInfo::Component::Genre::Automatic) {
          Descriptor &toDesc{
              *reinterpret_cast<Descriptor *>(toPtr + component->offset)};
          if (toDesc.raw().base_addr != nullptr) {
            toDesc.set_base_addr(nullptr);
            RUNTIME_CHECK(terminator, toDesc.Allocate() == CFI_SUCCESS);
            const Descriptor &fromDesc{*reinterpret_cast<const Descriptor *>(
                fromPtr + component->offset)};
            CopyArray(toDesc, fromDesc, terminator);
          }
        }
      }
    }
  }
}

void CopyArray(
    const Descriptor &to, const Descriptor &from, Terminator &terminator) {
  std::size_t elements{to.Elements()};
  RUNTIME_CHECK(terminator, elements == from.Elements());
  SubscriptValue toAt[maxRank], fromAt[maxRank];
  to.GetLowerBounds(toAt);
  from.GetLowerBounds(fromAt);
  while (elements-- > 0) {
    CopyElement(to, toAt, from, fromAt, terminator);
    to.IncrementSubscripts(toAt);
    from.IncrementSubscripts(fromAt);
  }
}
} // namespace Fortran::runtime
