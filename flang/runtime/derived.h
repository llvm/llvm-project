//===-- runtime/derived.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Internal runtime utilities for derived type operations.

#ifndef FLANG_RUNTIME_DERIVED_H_
#define FLANG_RUNTIME_DERIVED_H_

namespace Fortran::runtime::typeInfo {
class DerivedType;
}

namespace Fortran::runtime {
class Descriptor;
class Terminator;

// Perform default component initialization, allocate automatic components.
// Returns a STAT= code (0 when all's well).
int Initialize(const Descriptor &, const typeInfo::DerivedType &, Terminator &,
    bool hasStat = false, const Descriptor *errMsg = nullptr);

// Call FINAL subroutines, deallocate allocatable & automatic components.
// Does not deallocate the original descriptor.
void Destroy(const Descriptor &, bool finalize, const typeInfo::DerivedType &);

// Assigns one object to another via intrinsic assignment (F'2018 10.2.1.3) or
// defined assignment (10.2.1.4), as appropriate.  Performs scalar expansion
// or allocatable reallocation as needed.  Does not perform intrinsic
// assignment implicit type conversion.
void Assign(Descriptor &, const Descriptor &, const typeInfo::DerivedType &,
    Terminator &);

} // namespace Fortran::runtime
#endif // FLANG_RUNTIME_DERIVED_H_
