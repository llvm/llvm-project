//===-- Unittests for SbrkHeap --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// sbrk_heap.h must be included before Test.h so operator delete attributes
// in new.h apply before any test framework declarations.
#include "test/UnitTest/sbrk_heap.h"

#include "test/UnitTest/Test.h"

asm(R"(
.globl _end, __llvm_libc_heap_limit

.bss
_end:
  .fill 1024
__llvm_libc_heap_limit:
)");

TEST(LlvmLibcSbrkHeapTest, BasicAllocationAndDoubling) {
  // Start with a very small initial heap (e.g. 512 bytes)
  LIBC_NAMESPACE::SbrkHeap heap(512);

  // Allocate 300 bytes; this should fit in the initial 512-byte heap.
  void *ptr1 = heap.allocate(300);
  EXPECT_NE(ptr1, static_cast<void *>(nullptr));

  // Allocate another 400 bytes; this exceeds the initial 512-byte heap,
  // triggering SYS_brk growth (doubling the heap by adopting another 512
  // bytes).
  void *ptr2 = heap.allocate(400);
  EXPECT_NE(ptr2, static_cast<void *>(nullptr));

  // Allocate a larger block (2048 bytes), triggering multiple doublings via
  // SYS_brk.
  void *ptr3 = heap.allocate(2048);
  EXPECT_NE(ptr3, static_cast<void *>(nullptr));

  heap.free(ptr1);
  heap.free(ptr2);
  heap.free(ptr3);
}

TEST(LlvmLibcSbrkHeapTest, ReallocAndCalloc) {
  LIBC_NAMESPACE::SbrkHeap heap(1024);

  void *ptr = heap.calloc(10, 100); // 1000 bytes
  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(static_cast<char *>(ptr)[i], char(0));

  // Realloc to a larger size that requires growing the heap via SYS_brk.
  void *new_ptr = heap.realloc(ptr, 3000);
  ASSERT_NE(new_ptr, static_cast<void *>(nullptr));

  heap.free(new_ptr);
}

TEST(LlvmLibcSbrkHeapTest, MergeAcrossAdoptedRegion) {
  LIBC_NAMESPACE::SbrkHeap heap(512);

  void *ptr1 = heap.allocate(300);
  EXPECT_NE(ptr1, static_cast<void *>(nullptr));

  void *ptr2 = heap.allocate(400);
  EXPECT_NE(ptr2, static_cast<void *>(nullptr));

  heap.free(ptr1);
  heap.free(ptr2);

  size_t size_before = heap.region().size();

  void *ptr3 = heap.allocate(1100);
  EXPECT_NE(ptr3, static_cast<void *>(nullptr));

  size_t size_after = heap.region().size();

  EXPECT_EQ(size_after, size_before);

  heap.free(ptr3);
}
