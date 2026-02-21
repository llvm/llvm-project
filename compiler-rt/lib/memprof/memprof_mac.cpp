//===-- memprof_mac.cpp ---------------------------------------------------===//
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
#include "memprof_mapping.h"
#include "memprof_stack.h"
#include "memprof_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mac.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <libkern/OSAtomic.h>
#include <mach-o/dyld.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#include <unistd.h>

namespace __memprof {

void InitializePlatformInterceptors() {}
void InitializePlatformExceptionHandlers() {}

uptr FindDynamicShadowStart() {
  uptr shadow_size_bytes = MemToShadowSize(kHighMemEnd);
  return MapDynamicShadow(shadow_size_bytes, SHADOW_SCALE,
                          /*min_shadow_base_alignment*/ 0, kHighMemEnd,
                          GetMmapGranularity());
}

void *MemprofDlSymNext(const char *sym) { return dlsym(RTLD_NEXT, sym); }

} // namespace __memprof

#endif // SANITIZER_APPLE
