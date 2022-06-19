//===-- include/flang/Runtime/assign.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External and internal APIs for data assignment (both intrinsic assignment
// and TBP defined generic ASSIGNMENT(=)).  Should be called by lowering
// for any assignments possibly needing special handling.  Intrinsic assignment
// to non-allocatable variables whose types are intrinsic need not come through
// here (though they may do so).  Assignments to allocatables, and assignments
// whose types may be polymorphic or are monomorphic and of derived types with
// finalization, allocatable components, or components with type-bound defined
// assignments, in the original type or the types of its non-pointer components
// (recursively) must arrive here.
//
// Non-type-bound generic INTERFACE ASSIGNMENT(=) is resolved in semantics and
// need not be handled here in the runtime; ditto for type conversions on
// intrinsic assignments.

#ifndef FORTRAN_RUNTIME_ASSIGN_H_
#define FORTRAN_RUNTIME_ASSIGN_H_

#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
class Descriptor;
class Terminator;

// Assigns one object to another via intrinsic assignment (F'2018 10.2.1.3) or
// type-bound (only!) defined assignment (10.2.1.4), as appropriate.  Performs
// finalization, scalar expansion, & allocatable (re)allocation as needed.
// Does not perform intrinsic assignment implicit type conversion.  Both
// descriptors must be initialized.  Recurses as needed to handle components.
void Assign(Descriptor &, const Descriptor &, Terminator &);

extern "C" {
// API for lowering assignment
void RTNAME(Assign)(Descriptor &to, const Descriptor &from,
    const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_H_
