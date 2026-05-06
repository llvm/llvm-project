//===-- include/flang/Runtime/assign.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External APIs for data assignment (both intrinsic assignment and TBP defined
// generic ASSIGNMENT(=)).  Should be called by lowering for any assignments
// possibly needing special handling.  Intrinsic assignment to non-allocatable
// variables whose types are intrinsic need not come through here (though they
// may do so).  Assignments to allocatables, and assignments whose types may be
// polymorphic or are monomorphic and of derived types with finalization,
// allocatable components, or components with type-bound defined assignments, in
// the original type or the types of its non-pointer components (recursively)
// must arrive here.
//
// Non-type-bound generic INTERFACE ASSIGNMENT(=) is resolved in semantics and
// need not be handled here in the runtime apart from derived type components;
// ditto for type conversions on intrinsic assignments.

#ifndef FORTRAN_RUNTIME_ASSIGN_H_
#define FORTRAN_RUNTIME_ASSIGN_H_

#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/freestanding-tools.h"

namespace Fortran::runtime {
class Descriptor;
class Terminator;

enum AssignFlags {
  NoAssignFlags = 0,
  MaybeReallocate = 1 << 0,
  NeedFinalization = 1 << 1,
  CanBeDefinedAssignment = 1 << 2,
  ComponentCanBeDefinedAssignment = 1 << 3,
  ExplicitLengthCharacterLHS = 1 << 4,
  PolymorphicLHS = 1 << 5,
  DeallocateLHS = 1 << 6,
  UpdateLHSBounds = 1 << 7,
};

#ifdef RT_DEVICE_COMPILATION
RT_API_ATTRS void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, MemmoveFct = &MemmoveWrapper);
#else
RT_API_ATTRS void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, MemmoveFct = &runtime::memmove);
#endif

extern "C" {

// API for lowering assignment
void RTDECL(Assign)(Descriptor &to, const Descriptor &from,
    const char *sourceFile = nullptr, int sourceLine = 0);
// This variant has no finalization, defined assignment, or allocatable
// reallocation.
void RTDECL(AssignTemporary)(Descriptor &to, const Descriptor &from,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Establish "temp" descriptor as an allocatable descriptor with the same type,
// rank, and length parameters as "var" and copy "var" to it using
// AssignTemporary.
void RTDECL(CopyInAssign)(Descriptor &temp, const Descriptor &var,
    const char *sourceFile = nullptr, int sourceLine = 0);
// When "var" is provided, copy "temp" to it assuming "var" is already
// initialized. Destroy and deallocate "temp" in all cases.
void RTDECL(CopyOutAssign)(Descriptor *var, Descriptor &temp,
    const char *sourceFile = nullptr, int sourceLine = 0);
// This variant is for assignments to explicit-length CHARACTER left-hand
// sides that might need to handle truncation or blank-fill, and
// must maintain the character length even if an allocatable array
// is reallocated.
void RTDECL(AssignExplicitLengthCharacter)(Descriptor &to,
    const Descriptor &from, const char *sourceFile = nullptr,
    int sourceLine = 0);
// This variant is assignments to whole polymorphic allocatables.
void RTDECL(AssignPolymorphic)(Descriptor &to, const Descriptor &from,
    const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_H_
