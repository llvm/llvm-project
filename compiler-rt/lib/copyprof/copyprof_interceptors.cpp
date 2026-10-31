//===-- copyprof_interceptors.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements standard library function interceptors for CopyProf,
/// tracking memory operations across intercepted library calls.
///
//===----------------------------------------------------------------------===//

#include "copyprof_interceptors.h"

#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_internal.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

using namespace __copyprof;

namespace __copyprof {

int CopyProfOnExit() { return 0; }

}  // namespace __copyprof

#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...) \
  do {                                           \
    ctx = 0;                                     \
    (void)ctx;                                   \
    if (copyprof_init_is_running)                \
      return REAL(func)(__VA_ARGS__);            \
    if (UNLIKELY(!copyprof_is_initialized))      \
      __copyprof_init_once();                    \
  } while (false)

#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  do {                                                \
    (void)(ctx);                                      \
    (void)(ptr);                                      \
    (void)(size);                                     \
  } while (false)

#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  __copyprof_store_callback(ptr, size)

#define COMMON_INTERCEPTOR_DIR_ACQUIRE(ctx, path) \
  do {                                            \
  } while (false)

#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
  do {                                         \
  } while (false)

#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
  do {                                         \
  } while (false)

#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) \
  do {                                                      \
  } while (false)

#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) \
  do {                                                \
  } while (false)

#define COMMON_INTERCEPTOR_ON_EXIT(ctx) CopyProfOnExit()

#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  do {                                                         \
  } while (false)

#define COMMON_INTERCEPT_FUNCTION_VER(name, ver)                          \
  do {                                                                    \
    if (!INTERCEPT_FUNCTION_VER(name, ver))                               \
      VReport(1, "CopyProf: failed to intercept '%s@@%s'\n", #name, ver); \
  } while (0)

#define COMMON_INTERCEPT_FUNCTION_VER_UNVERSIONED_FALLBACK(name, ver)       \
  do {                                                                      \
    if (!INTERCEPT_FUNCTION_VER(name, ver) && !INTERCEPT_FUNCTION(name))    \
      VReport(1, "CopyProf: failed to intercept '%s@@%s' or '%s'\n", #name, \
              ver, #name);                                                  \
  } while (0)

#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)

#define COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED (!copyprof_is_initialized)

#include "sanitizer_common/sanitizer_platform_interceptors.h"
// These interceptors allocate memory via `WRAP(malloc)`, but heap functions are
// not intercepted yet.
#undef SANITIZER_INTERCEPT_STRNDUP
#undef SANITIZER_INTERCEPT___STRNDUP
#undef SANITIZER_INTERCEPT_REALPATH

#include "sanitizer_common/sanitizer_common_interceptors.inc"

namespace __copyprof {

void InitializeInterceptors() {
  static bool initialized;
  CHECK(!initialized);
  initialized = true;
  InitializeCommonInterceptors();
}

}  // namespace __copyprof
