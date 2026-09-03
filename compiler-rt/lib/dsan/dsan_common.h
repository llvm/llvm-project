//=-- dsan_common.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Private DSan header.
//
//===----------------------------------------------------------------------===//

#ifndef DSAN_COMMON_H
#define DSAN_COMMON_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_range.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_thread_registry.h"

// DoubleFreeSanitizer can run on most platforms that support sanitizers.
#if SANITIZER_ANDROID && (__ANDROID_API__ < 28 || defined(__arm__))
#  define CAN_SANITIZE_DOUBLE_FREE 0
#elif (SANITIZER_LINUX || SANITIZER_APPLE) && (SANITIZER_WORDSIZE == 64) && \
    (defined(__x86_64__) || defined(__mips64) || defined(__aarch64__) ||    \
     defined(__powerpc64__) || defined(__s390x__))
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif defined(__i386__) && (SANITIZER_LINUX || SANITIZER_APPLE)
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif defined(__arm__) && SANITIZER_LINUX
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif defined(__hexagon__) && SANITIZER_LINUX
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif SANITIZER_LOONGARCH64 && SANITIZER_LINUX
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif SANITIZER_RISCV64 && SANITIZER_LINUX
#  define CAN_SANITIZE_DOUBLE_FREE 1
#elif SANITIZER_NETBSD || SANITIZER_FUCHSIA
#  define CAN_SANITIZE_DOUBLE_FREE 1
#else
#  define CAN_SANITIZE_DOUBLE_FREE 0
#endif

namespace __sanitizer {
class ThreadRegistry;
class ThreadContextBase;
struct DTLS;
}  // namespace __sanitizer

namespace __dsan {

// Returns true if [addr, addr + sizeof(void *)) is poisoned.
bool WordIsPoisoned(uptr addr);

//// --------------------------------------------------------------------------
//// Thread prototypes.
//// --------------------------------------------------------------------------

void LockThreads() SANITIZER_NO_THREAD_SAFETY_ANALYSIS;
void UnlockThreads() SANITIZER_NO_THREAD_SAFETY_ANALYSIS;
void EnsureMainThreadIDIsCorrect();

bool GetThreadRangesLocked(ThreadID os_id, uptr* stack_begin, uptr* stack_end,
                           uptr* tls_begin, uptr* tls_end, uptr* cache_begin,
                           uptr* cache_end, DTLS** dtls);
void GetAllThreadAllocatorCachesLocked(InternalMmapVector<uptr>* caches);
void GetThreadExtraStackRangesLocked(InternalMmapVector<Range>* ranges);
void GetThreadExtraStackRangesLocked(ThreadID os_id,
                                     InternalMmapVector<Range>* ranges);
void GetAdditionalThreadContextPtrsLocked(InternalMmapVector<uptr>* ptrs);
void GetRunningThreadsLocked(InternalMmapVector<ThreadID>* threads);
void PrintThreads();

//// --------------------------------------------------------------------------
//// Allocator prototypes.
//// --------------------------------------------------------------------------

void LockAllocator();
void UnlockAllocator();

void GetAllocatorCacheRange(uptr* begin, uptr* end);

void InitCommonDsan();

// Forward declaration - defined in dsan_thread.h
class ThreadContextDsanBase;

ThreadContextDsanBase* GetCurrentThread();
void SetCurrentThread(ThreadContextDsanBase* tctx);

}  // namespace __dsan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE const char*
__dsan_default_options();

}  // extern "C"

#endif  // DSAN_COMMON_H
