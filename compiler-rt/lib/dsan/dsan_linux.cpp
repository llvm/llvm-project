//=-- dsan_linux.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer. Linux/NetBSD/Fuchsia-specific
// code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX || SANITIZER_NETBSD || SANITIZER_FUCHSIA

#  include "dsan_allocator.h"
#  include "dsan_thread.h"

namespace __dsan {

static THREADLOCAL ThreadContextDsanBase* current_thread = nullptr;
ThreadContextDsanBase* GetCurrentThread() { return current_thread; }
void SetCurrentThread(ThreadContextDsanBase* tctx) { current_thread = tctx; }

static THREADLOCAL AllocatorCache allocator_cache;
AllocatorCache* GetAllocatorCache() { return &allocator_cache; }

void ReplaceSystemMalloc() {}

}  // namespace __dsan

#endif  // SANITIZER_LINUX || SANITIZER_NETBSD || SANITIZER_FUCHSIA
