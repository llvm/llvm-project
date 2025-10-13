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

/// Called when a `-fbounds-safety` bounds check fails when building with
/// `-fbounds-safety-soft-traps=call-with-str`. This function is allowed to
/// to return.
///
/// \param reason A string constant describing the reason for trapping or
///        NULL.
void __bounds_safety_soft_trap_s(const char *reason);

// TODO(dliew): Document the `reason_code` values (rdar://162824128)

/// Called when a `-fbounds-safety` bounds check fails when building with
/// `-fbounds-safety-soft-traps=call-with-code`. This function is allowed to
/// to return.
///
/// \param reason_code. An integer the represents the reason for trapping.
///        The values are currently not documented but will be in the future.
///
void __bounds_safety_soft_trap_c(uint16_t reason_code);

#ifdef __cplusplus
}
#endif

#endif
