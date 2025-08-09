//===-- lib/runtime/support.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/support.h"
#include "ISO_Fortran_util.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/type-info.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

bool RTDEF(IsContiguous)(const Descriptor &descriptor) {
  return descriptor.IsContiguous();
}

bool RTDEF(IsContiguousUpTo)(const Descriptor &descriptor, int dim) {
  return descriptor.IsContiguous(dim);
}

bool RTDEF(IsAssumedSize)(const Descriptor &descriptor) {
  return ISO::IsAssumedSize(&descriptor.raw());
}

void RTDEF(CopyAndUpdateDescriptor)(Descriptor &to, const Descriptor &from,
    const typeInfo::DerivedType *newDynamicType,
    ISO::CFI_attribute_t newAttribute, enum LowerBoundModifier newLowerBounds) {
  to = from;
  if (newDynamicType) {
    DescriptorAddendum *toAddendum{to.Addendum()};
    INTERNAL_CHECK(toAddendum);
    toAddendum->set_derivedType(newDynamicType);
    to.raw().elem_len = newDynamicType->sizeInBytes();
  }
  to.raw().attribute = newAttribute;
  if (newLowerBounds != LowerBoundModifier::Preserve) {
    const ISO::CFI_index_t newLowerBound{
        newLowerBounds == LowerBoundModifier::SetToOnes ? 1 : 0};
    const int rank{to.rank()};
    for (int i = 0; i < rank; ++i) {
      to.GetDimension(i).SetLowerBound(newLowerBound);
    }
  }
}

void *RTDEF(DescriptorGetBaseAddress)(
    const Descriptor &desc, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  void *baseAddr = desc.raw().base_addr;
  if (!baseAddr) {
    terminator.Crash("Could not retrieve Descriptor's base address");
  }
  return baseAddr;
}

std::size_t RTDEF(DescriptorGetDataSizeInBytes)(
    const Descriptor &desc, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  std::size_t descElements{desc.Elements()};
  if (!descElements) {
    terminator.Crash("Could not retrieve Descriptor's Elements");
  }
  std::size_t descElementBytes{desc.ElementBytes()};
  if (!descElementBytes) {
    terminator.Crash("Could not retrieve Descriptor's ElementBytes");
  }
  return descElements * descElementBytes;
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
