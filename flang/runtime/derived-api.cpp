//===-- runtime/derived-api.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/derived-api.h"
#include "derived.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

extern "C" {

void RTNAME(Initialize)(
    const Descriptor &descriptor, const char *sourceFile, int sourceLine) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noInitializationNeeded()) {
        Terminator terminator{sourceFile, sourceLine};
        Initialize(descriptor, *derived, terminator);
      }
    }
  }
}

void RTNAME(Destroy)(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noDestructionNeeded()) {
        Destroy(descriptor, true, *derived);
      }
    }
  }
}

bool RTNAME(ClassIs)(
    const Descriptor &descriptor, const typeInfo::DerivedType &derivedType) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (derived == &derivedType) {
        return true;
      }
      const typeInfo::DerivedType *parent{derived->GetParentType()};
      while (parent) {
        if (parent == &derivedType) {
          return true;
        }
        parent = parent->GetParentType();
      }
    }
  }
  return false;
}

// TODO: Assign()

} // extern "C"
} // namespace Fortran::runtime
