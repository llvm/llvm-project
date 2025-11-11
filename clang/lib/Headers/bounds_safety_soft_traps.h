/*===---- bounds_safety_soft_traps.h ----------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
  This file defines the interface used by `-fbounds-safety`'s trap mode. Note
  this interface isn't yet considered stable
 *===------------------------------------------------------------------------===
 */

#ifndef __CLANG_BOUNDS_SAFETY_SOFT_TRAPS_H
#define __CLANG_BOUNDS_SAFETY_SOFT_TRAPS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Macro that defines the current API version. This value can be queried at
/// compile time to know which interface version the compiler uses.
#define __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION 0

// `__CLANG_BOUNDS_SAFETY_SOFT_TRAP_FN_ATTRS` is guarded because `preserve_all`
// is only implemented for x86_64 and arm64. For other targets we get a
// `'preserve_all' calling convention is not supported for this target` warning.
#if defined(__aarch64__) || defined(__x86_64__)
// This attribute is used to try to reduce the codesize impact of making calls
// to the soft trap runtime function(s).
#define __CLANG_BOUNDS_SAFETY_SOFT_TRAP_FN_ATTRS __attribute__((preserve_all))
#else
#define __CLANG_BOUNDS_SAFETY_SOFT_TRAP_FN_ATTRS
#endif

/// Called when a `-fbounds-safety` bounds check fails when building with
/// `-fbounds-safety-soft-traps=call-with-str`. This function is allowed to
/// to return.
///
/// \param reason A string constant describing the reason for trapping or
///        NULL.
__CLANG_BOUNDS_SAFETY_SOFT_TRAP_FN_ATTRS
void __bounds_safety_soft_trap_s(const char *reason);

/// Called when a `-fbounds-safety` bounds check fails when building with
/// `-fbounds-safety-soft-traps=call-with-code`. This function is allowed to
/// to return. This function takes no arguments.
__CLANG_BOUNDS_SAFETY_SOFT_TRAP_FN_ATTRS
void __bounds_safety_soft_trap(void);

#ifdef __cplusplus
}
#endif

#endif
