//===-- memprof_mac.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// Mac-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_APPLE

#include "memprof_interceptors.h"
#include "memprof_internal.h"
#include "memprof_thread.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_freebsd.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <unwind.h>

namespace __memprof {

void InitializePlatformInterceptors() {}
void InitializePlatformExceptionHandlers() {}

// No-op. Mac does not support static linkage anyway.
void *MemprofDoesNotSupportStaticLinkage() {
  return 0;
}

uptr FindDynamicShadowStart() {
  uptr shadow_size_bytes = MemToShadowSize(kHighMemEnd);
  return MapDynamicShadow(shadow_size_bytes, SHADOW_SCALE,
                          /*min_shadow_base_alignment*/ 0, kHighMemEnd);
}

void *MemprofDlSymNext(const char *sym) { return dlsym(RTLD_NEXT, sym); }

} // namespace __memprof
#endif
