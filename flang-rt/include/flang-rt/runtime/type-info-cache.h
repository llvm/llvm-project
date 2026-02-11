//===-- include/flang-rt/runtime/type-info-cache.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cache for concrete PDT (Parameterized Derived Type) instantiations.
//
// For types with LEN parameters, layout depends on runtime values. This cache
// stores resolved "concrete types" keyed by (generic_type, len_values...)
// so that all instances with identical LEN values share the same type
// description.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_TYPE_INFO_CACHE_H_
#define FLANG_RT_RUNTIME_TYPE_INFO_CACHE_H_

#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/type-info.h"

namespace Fortran::runtime {
class Terminator;
} // namespace Fortran::runtime

namespace Fortran::runtime::typeInfo {

// Get the concrete type for a generic PDT instantiated with specific LEN
// values.
//
// If the type has no LEN parameters, returns the generic type unchanged.
// Otherwise, looks up or creates a concrete type with resolved offsets/sizes.
//
// Parameters:
//   genericType - The uninstantiated (generic) DerivedType from compile time
//   instance    - A descriptor whose addendum contains the actual LEN values
//   terminator  - For error handling
//
// Returns:
//   Pointer to concrete DerivedType (may be same as genericType if no LEN
//   params)
//
RT_API_ATTRS const DerivedType *GetConcreteType(const DerivedType &genericType,
    const Descriptor &instance, runtime::Terminator &terminator);

} // namespace Fortran::runtime::typeInfo

#endif // FLANG_RT_RUNTIME_TYPE_INFO_CACHE_H_
