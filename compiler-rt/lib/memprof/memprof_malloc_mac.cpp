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
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

// ---------------------- Replacement functions ---------------- {{{1
using namespace __memprof;

struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return memprof_init_is_running; }
};

namespace __memprof {
void ReplaceSystemMalloc() {}
} // namespace __memprof

#endif
