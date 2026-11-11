//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Death tests for freelist_heap.
///
//===----------------------------------------------------------------------===//
#include "src/__support/freelist_heap.h"
#include "test/UnitTest/Test.h"

// Just a stub, not really used
asm(".globl _end, __llvm_libc_heap_limit\n_end:\n__llvm_libc_heap_limit:\n");

using LIBC_NAMESPACE::FreeListHeap;
using LIBC_NAMESPACE::cpp::byte;

TEST(LlvmLibcFreeListHeapDeathTest, DoubleFreeDeath) {
  byte buf[2048] = {byte(0)};
  FreeListHeap allocator(buf);
  void *ptr = allocator.allocate(512);
  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  allocator.free(ptr);
#if defined(ENABLE_SUBPROCESS_TESTS) &&                                        \
    LIBC_COPT_HARDENING_MODE > LIBC_HARDENING_MODE_FAST
  EXPECT_EXITS(
      [&] {
        allocator.free(ptr);
        allocator.integrity_check();
      },
      255);
#endif
}

TEST(LlvmLibcFreeListHeapDeathTest, UseAfterFreeDeath) {
  byte buf[2048] = {byte(0)};
  FreeListHeap allocator(buf);
  void *ptr = allocator.allocate(512);
  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  allocator.free(ptr);
#if defined(ENABLE_SUBPROCESS_TESTS) &&                                        \
    LIBC_COPT_HARDENING_MODE > LIBC_HARDENING_MODE_FAST
  new (ptr) uintptr_t{0xDEADBEEF};
  EXPECT_EXITS([&] { allocator.integrity_check(); }, 255);
#endif
}
