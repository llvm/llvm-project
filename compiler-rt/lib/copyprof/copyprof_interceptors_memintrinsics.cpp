//===-- copyprof_interceptors_memintrinsics.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements interceptors for memory intrinsic functions (memcpy,
/// memmove, memset) to correctly maintain shadow memory status.
///
//===----------------------------------------------------------------------===//

#define SANITIZER_COMMON_NO_REDEFINE_BUILTINS

#include "copyprof_interceptors_memintrinsics.h"

#include "copyprof_interceptors.h"
#include "copyprof_interface_internal.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __copyprof;
using namespace __sanitizer;

#define COPYPROF_INTERCEPTOR_ENTER(ctx, func) \
  ctx = 0;                                    \
  (void)ctx;

// The intrinsics may be called early during startup so CopyProf itself may not
// be initialized yet (f.e. the intrinsics may be called during
// `__copyprof_init_once` from the internals of `Printf`).
#define COPYPROF_MEMCPY_IMPL(to, from, size)               \
  do {                                                     \
    if (UNLIKELY(!copyprof_is_initialized))                \
      return __sanitizer::internal_memcpy(to, from, size); \
    COPYPROF_WRITE_RANGE(to, size);                        \
    return REAL(memcpy)(to, from, size);                   \
  } while (0)

#define COPYPROF_MEMSET_IMPL(block, c, size)  \
  do {                                        \
    if (UNLIKELY(!copyprof_is_initialized))   \
      return internal_memset(block, c, size); \
    COPYPROF_WRITE_RANGE(block, size);        \
    return REAL(memset)(block, c, size);      \
  } while (0)

#define COPYPROF_MEMMOVE_IMPL(to, from, size)               \
  do {                                                      \
    if (UNLIKELY(!copyprof_is_initialized))                 \
      return __sanitizer::internal_memmove(to, from, size); \
    COPYPROF_WRITE_RANGE(to, size);                         \
    return REAL(memmove)(to, from, size);                   \
  } while (0)

#define COMMON_INTERCEPTOR_MEMCPY_IMPL(ctx, to, from, size) \
  do {                                                      \
    COPYPROF_INTERCEPTOR_ENTER(ctx, memcpy);                \
    COPYPROF_MEMCPY_IMPL(to, from, size);                   \
  } while (false)

#define COMMON_INTERCEPTOR_MEMSET_IMPL(ctx, block, c, size) \
  do {                                                      \
    COPYPROF_INTERCEPTOR_ENTER(ctx, memset);                \
    COPYPROF_MEMSET_IMPL(block, c, size);                   \
  } while (false)

#define COMMON_INTERCEPTOR_MEMMOVE_IMPL(ctx, block, c, size) \
  do {                                                       \
    COPYPROF_INTERCEPTOR_ENTER(ctx, memmove);                \
    COPYPROF_MEMMOVE_IMPL(block, c, size);                   \
  } while (false)

// Needed by the default fortified _chk implementations, which cannot use REAL()
// until initialization has completed.
#define COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED (!copyprof_is_initialized)

#include "sanitizer_common/sanitizer_common_interceptors_memintrinsics.inc"
