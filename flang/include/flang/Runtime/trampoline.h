//===-- include/flang/Runtime/trampoline.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime support for W^X-compliant trampoline pool management.
//
// This provides an alternative to stack-based trampolines for internal
// procedures with host association. Instead of requiring the stack to be
// both writable and executable (violating W^X security policies), this
// implementation uses a pool of trampoline stubs generated during pool
// initialization in a separate executable (but not writable) memory region,
// paired with writable (but not executable) data entries.
//
// See flang/docs/InternalProcedureTrampolines.md for design details.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TRAMPOLINE_H_
#define FORTRAN_RUNTIME_TRAMPOLINE_H_

#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
extern "C" {

/// Initializes a new trampoline and returns its internal handle.
///
/// Allocates a trampoline entry from the pool, configuring it to call
/// \p calleeAddress with the static chain pointer \p staticChainAddress
/// set in the appropriate register (per target ABI).
///
/// \p scratch is currently ignored, and Flang lowering passes nullptr.
///
/// The returned handle must be passed to TrampolineFree() when the
/// host procedure exits.
///
/// Pool capacity is fixed after initialization: 1024 slots by default,
/// configurable via FLANG_TRAMPOLINE_POOL_SIZE. Allocating from a full pool
/// issues a fatal error.
///
/// Architecture support: Currently x86-64 and AArch64. On unsupported
/// architectures, calling this function issues a fatal diagnostic.
void *RTDECL(TrampolineInit)(
    void *scratch, const void *calleeAddress, const void *staticChainAddress);

/// Returns the callable trampoline address for the given handle.
///
/// \p handle is a value returned by TrampolineInit().
/// The result is a function pointer that can be called directly; it will
/// set up the static chain register and jump to the original callee.
void *RTDECL(TrampolineAdjust)(void *handle);

/// Frees the trampoline entry associated with the given handle.
///
/// Must be called at every exit from the host procedure to return the
/// trampoline slot to the pool. After this call, any function pointer
/// previously obtained via TrampolineAdjust() for this handle becomes
/// invalid.
void RTDECL(TrampolineFree)(void *handle);

} // extern "C"
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_TRAMPOLINE_H_
