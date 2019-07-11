//=-- lsan_linux.cc -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer. Linux/NetBSD-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX || SANITIZER_NETBSD

#include "lsan_allocator.h"

namespace __lsan {

static THREADLOCAL u32 current_thread_tid = kInvalidTid;
u32 GetCurrentThread() { return current_thread_tid; }
void SetCurrentThread(u32 tid) { current_thread_tid = tid; }

static THREADLOCAL AllocatorCache allocator_cache;
AllocatorCache *GetAllocatorCache() { return &allocator_cache; }

void ReplaceSystemMalloc() {}

} // namespace __lsan

#endif  // SANITIZER_LINUX || SANITIZER_NETBSD
