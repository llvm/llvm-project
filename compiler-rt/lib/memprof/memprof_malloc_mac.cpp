//===-- memprof_malloc_mac.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// Mac-specific malloc interception.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_APPLE

#include "memprof_allocator.h"
#include "memprof_interceptors.h"
#include "memprof_internal.h"
#include "memprof_stack.h"

using namespace __memprof;
#define COMMON_MALLOC_ZONE_NAME "memprof"
#define COMMON_MALLOC_ENTER()                                                  \
  do {                                                                         \
    MemprofInitFromRtl();                                                      \
  } while (false)
#define COMMON_MALLOC_SANITIZER_INITIALIZED memprof_inited
#define COMMON_MALLOC_FORCE_LOCK() memprof_mz_force_lock()
#define COMMON_MALLOC_FORCE_UNLOCK() memprof_mz_force_unlock()
#define COMMON_MALLOC_MEMALIGN(alignment, size)                                \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *p = memprof_memalign(alignment, size, &stack, FROM_MALLOC)
#define COMMON_MALLOC_MALLOC(size)                                             \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *p = memprof_malloc(size, &stack)
#define COMMON_MALLOC_REALLOC(ptr, size)                                       \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *p = memprof_realloc(ptr, size, &stack);
#define COMMON_MALLOC_CALLOC(count, size)                                      \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *p = memprof_calloc(count, size, &stack);
#define COMMON_MALLOC_POSIX_MEMALIGN(memptr, alignment, size)                  \
  GET_STACK_TRACE_MALLOC;                                                      \
  int res = memprof_posix_memalign(memptr, alignment, size, &stack);
#define COMMON_MALLOC_VALLOC(size)                                             \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *p = memprof_memalign(GetPageSizeCached(), size, &stack, FROM_MALLOC);
#define COMMON_MALLOC_FREE(ptr)                                                \
  GET_STACK_TRACE_FREE;                                                        \
  memprof_free(ptr, &stack, FROM_MALLOC);
#define COMMON_MALLOC_SIZE(ptr) uptr size = memprof_mz_size(ptr);
#define COMMON_MALLOC_FILL_STATS(zone, stats)
#define COMMON_MALLOC_REPORT_UNKNOWN_REALLOC(ptr, zone_ptr, zone_name)
#define COMMON_MALLOC_NAMESPACE __memprof
#define COMMON_MALLOC_HAS_ZONE_ENUMERATOR 0
#define COMMON_MALLOC_HAS_EXTRA_INTROSPECTION_INIT 0

#include "sanitizer_common/sanitizer_malloc_mac.inc"

namespace COMMON_MALLOC_NAMESPACE {

bool HandleDlopenInit() {
  static_assert(SANITIZER_SUPPORTS_INIT_FOR_DLOPEN,
                "Expected SANITIZER_SUPPORTS_INIT_FOR_DLOPEN to be true");
  auto init_str = GetEnv("APPLE_MEMPROF_INIT_FOR_DLOPEN");
  if (!init_str)
    return false;
  if (internal_strncmp(init_str, "1", 1) != 0)
    return false;
  InitMallocZoneFields();
  return true;
}

} // namespace COMMON_MALLOC_NAMESPACE

#endif // SANITIZER_APPLE
