//===-- runtime/assign.h-----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Internal APIs for data assignment (both intrinsic assignment and TBP defined
// generic ASSIGNMENT(=)).

#ifndef FORTRAN_RUNTIME_ASSIGN_INTERNAL_H_
#define FORTRAN_RUNTIME_ASSIGN_INTERNAL_H_

namespace Fortran::runtime {
class Descriptor;
class Terminator;

// Assigns one object to another via intrinsic assignment (F'2018 10.2.1.3) or
// type-bound (only!) defined assignment (10.2.1.4), as appropriate.  Performs
// finalization, scalar expansion, & allocatable (re)allocation as needed.
// Does not perform intrinsic assignment implicit type conversion.  Both
// descriptors must be initialized.  Recurses as needed to handle components.
// Do not perform allocatable reallocation if \p skipRealloc is true, which is
// used for allocate statement with source specifier.
void Assign(
    Descriptor &, const Descriptor &, Terminator &, bool skipRealloc = false);

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_INTERNAL_H_
