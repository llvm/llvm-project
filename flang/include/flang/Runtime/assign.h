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
#include <cstdint>

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

// Support for skipping copy-out into read-only memory
// (FLANG_RT_COPYOUT_READONLY_MODE; see flang/docs/RuntimeEnvironment.md).
// These entry points let the compiler apply the same policy in inlined
// copy-out code that CopyOutAssign applies internally. On device
// compilations they are stubs (mode 0 / false / no-op).
//
// Returns the mode: 0 = off (default), 1 = trust a one-time snapshot of the
// process memory map (no system calls), 2 = additionally re-confirm each
// snapshot hit against the current map. Parsed lazily from the environment;
// invalid values read as 0.
std::int32_t RTDECL(CopyOutReadOnlyMode)();
// True iff the whole data span of the descriptor lies within the snapshot's
// read-only regions. No system calls after the one-time lazy snapshot.
// Meaningful only when the mode is nonzero; fail-closed (false) on any
// uncertainty.
bool RTDECL(CopyOutReadOnlyCandidate)(const Descriptor &);
// True iff the whole data span of the descriptor is mapped read-only in the
// *current* process memory map. Performs system calls; intended as the mode-2
// re-confirmation of a candidate hit before skipping a copy-out.
bool RTDECL(CopyOutReadOnlyConfirm)(const Descriptor &);
// Diagnostics hook: count a skipped copy-out and, when
// FLANG_RT_COPYOUT_READONLY_DIAG=1, report the first few on stderr. Call it
// whenever inlined code skips a copy-out because of the checks above.
void RTDECL(NoteSkippedCopyOut)(
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
// Fast path for simple intrinsic type assignments (no derived types, no
// finalization)
void RTDECL(AssignSimple)(Descriptor &to, const Descriptor &from,
    const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_H_
