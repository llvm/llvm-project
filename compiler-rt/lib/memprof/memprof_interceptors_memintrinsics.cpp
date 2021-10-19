//===-- memprof_interceptors_memintrinsics.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// MemProf versions of memcpy, memmove, and memset.
//===---------------------------------------------------------------------===//

#include "memprof_interceptors_memintrinsics.h"
#include "memprof_stack.h"

using namespace __memprof;

void *__memprof_memcpy(void *to, const void *from, uptr size) {
  MEMPROF_MEMCPY_IMPL(to, from, size);
}

void *__memprof_memset(void *block, int c, uptr size) {
  MEMPROF_MEMSET_IMPL(block, c, size);
}

void *__memprof_memmove(void *to, const void *from, uptr size) {
  MEMPROF_MEMMOVE_IMPL(to, from, size);
}
