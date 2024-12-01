//===-- include/flang/Runtime/support.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for runtime support code for lowering.
#ifndef FORTRAN_RUNTIME_SUPPORT_H_
#define FORTRAN_RUNTIME_SUPPORT_H_

#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

namespace typeInfo {
class DerivedType;
}

enum class LowerBoundModifier : int {
  Preserve = 0,
  SetToOnes = 1,
  SetToZeroes = 2
};

extern "C" {

// Predicate: is the storage described by a Descriptor contiguous in memory?
bool RTDECL(IsContiguous)(const Descriptor &);

// Predicate: is this descriptor describing an assumed-size array?
bool RTDECL(IsAssumedSize)(const Descriptor &);

// Copy "from" descriptor into "to" descriptor and update "to" dynamic type,
// CFI_attribute, and lower bounds according to the other arguments.
// "newDynamicType" may be a null pointer in which case "to" dynamic type is the
// one of "from".
void RTDECL(CopyAndUpdateDescriptor)(Descriptor &to, const Descriptor &from,
    const typeInfo::DerivedType *newDynamicType,
    ISO::CFI_attribute_t newAttribute, enum LowerBoundModifier newLowerBounds);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_SUPPORT_H_
