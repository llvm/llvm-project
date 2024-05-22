//===-- runtime/derived.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Internal runtime utilities for derived type operations.

#ifndef FORTRAN_RUNTIME_DERIVED_H_
#define FORTRAN_RUNTIME_DERIVED_H_

#include "flang/Runtime/api-attrs.h"

namespace Fortran::runtime::typeInfo {
class DerivedType;
}

namespace Fortran::runtime {
class Descriptor;
class Terminator;

// Perform default component initialization, allocate automatic components.
// Returns a STAT= code (0 when all's well).
RT_API_ATTRS int Initialize(const Descriptor &, const typeInfo::DerivedType &,
    Terminator &, bool hasStat = false, const Descriptor *errMsg = nullptr);

// Call FINAL subroutines, if any
RT_API_ATTRS void Finalize(
    const Descriptor &, const typeInfo::DerivedType &derived, Terminator *);

// Call FINAL subroutines, deallocate allocatable & automatic components.
// Does not deallocate the original descriptor.
RT_API_ATTRS void Destroy(const Descriptor &, bool finalize,
    const typeInfo::DerivedType &, Terminator *);

// Return true if the passed descriptor is for a derived type
// entity that has a dynamic (allocatable, automatic) component.
RT_API_ATTRS bool HasDynamicComponent(const Descriptor &);

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_DERIVED_H_
