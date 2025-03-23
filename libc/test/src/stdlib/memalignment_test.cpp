//===-- Unittests for memalignment ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/memalignment.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

TEST(LlvmLibcMemAlignmentTest, NullPointer) {
  void *ptr = nullptr;
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr), static_cast<size_t>(0));
}

TEST(LlvmLibcMemAlignmentTest, MallocedPointers) {
  int *int_ptr = reinterpret_cast<int *>(LIBC_NAMESPACE::malloc(sizeof(int)));
  EXPECT_NE(reinterpret_cast<void *>(int_ptr), static_cast<void *>(nullptr));

  size_t int_alignment = LIBC_NAMESPACE::memalignment(int_ptr);
  EXPECT_GE(int_alignment, alignof(int));

  LIBC_NAMESPACE::free(int_ptr);

  // Allocate a double (typically 8-byte aligned)
  double *double_ptr =
      reinterpret_cast<double *>(LIBC_NAMESPACE::malloc(sizeof(double)));
  EXPECT_NE(reinterpret_cast<void *>(double_ptr), static_cast<void *>(nullptr));

  size_t double_alignment = LIBC_NAMESPACE::memalignment(double_ptr);
  EXPECT_GE(double_alignment, alignof(double));

  EXPECT_EQ(double_alignment & (double_alignment - 1), static_cast<size_t>(0));

  LIBC_NAMESPACE::free(double_ptr);
}

TEST(LlvmLibcMemAlignmentTest, SpecificAlignment) {

  // These addresses have known alignment patterns - if we can construct them
  uintptr_t addr_align2 = 0x2;   // 2-byte aligned
  uintptr_t addr_align4 = 0x4;   // 4-byte aligned
  uintptr_t addr_align8 = 0x8;   // 8-byte aligned
  uintptr_t addr_align16 = 0x10; // 16-byte aligned
  uintptr_t addr_align32 = 0x20; // 32-byte aligned

  void *ptr_align2 = reinterpret_cast<void *>(addr_align2);
  void *ptr_align4 = reinterpret_cast<void *>(addr_align4);
  void *ptr_align8 = reinterpret_cast<void *>(addr_align8);
  void *ptr_align16 = reinterpret_cast<void *>(addr_align16);
  void *ptr_align32 = reinterpret_cast<void *>(addr_align32);

  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_align2), static_cast<size_t>(2));
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_align4), static_cast<size_t>(4));
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_align8), static_cast<size_t>(8));
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_align16), static_cast<size_t>(16));
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_align32), static_cast<size_t>(32));

  uintptr_t addr_complex = 0x1234560; // 16-byte aligned (ends in 0)
  void *ptr_complex = reinterpret_cast<void *>(addr_complex);
  EXPECT_EQ(LIBC_NAMESPACE::memalignment(ptr_complex), static_cast<size_t>(16));
}

TEST(LlvmLibcMemAlignmentTest, AlignasSpecifiedAlignment) {
  alignas(16) static int aligned_16;
  alignas(32) static int aligned_32;
  alignas(64) static int aligned_64;
  alignas(128) static int aligned_128;
  alignas(256) static int aligned_256;

  EXPECT_GE(LIBC_NAMESPACE::memalignment(&aligned_16), static_cast<size_t>(16));
  EXPECT_GE(LIBC_NAMESPACE::memalignment(&aligned_32), static_cast<size_t>(32));
  EXPECT_GE(LIBC_NAMESPACE::memalignment(&aligned_64), static_cast<size_t>(64));
  EXPECT_GE(LIBC_NAMESPACE::memalignment(&aligned_128),
            static_cast<size_t>(128));
  EXPECT_GE(LIBC_NAMESPACE::memalignment(&aligned_256),
            static_cast<size_t>(256));
}
 