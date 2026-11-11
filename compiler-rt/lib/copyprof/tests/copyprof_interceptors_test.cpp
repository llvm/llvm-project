//===-- copyprof_interceptors_test.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_shadow.h"
#include "copyprof/copyprof_state.h"
#include "gtest/gtest.h"

namespace __copyprof {
namespace {

// The intercepted functions are called through volatile function pointers, so
// that the compiler cannot replace the calls with inline stores, which would
// bypass the interceptors.
void* (*volatile memset_fn)(void*, int, size_t) = memset;
void* (*volatile memcpy_fn)(void*, const void*, size_t) = memcpy;
void* (*volatile memmove_fn)(void*, const void*, size_t) = memmove;

class CopyProfInterceptorsTest : public testing::Test {
 protected:
  void SetUp() override {
    __copyprof_init_once();
    __copyprof_state = PerThreadState();
  }
};

TEST_F(CopyProfInterceptorsTest, MemsetMarksDestinationAsNonCopy) {
  alignas(8) char buf[32] = {0};
  MarkApplicationMemory(buf, sizeof(buf), /*is_copy=*/true);

  memset_fn(buf + 8, /*c=*/1, /*size=*/8);
  EXPECT_TRUE(IsMarkedAsCopy(buf, /*num_bytes=*/8));
  EXPECT_FALSE(IsMarkedAsCopy(buf + 8, /*num_bytes=*/8));
  EXPECT_TRUE(IsMarkedAsCopy(buf + 16, /*num_bytes=*/16));
}

TEST_F(CopyProfInterceptorsTest, MemcpyMarksOnlyDestinationAsNonCopy) {
  alignas(8) char dst[32] = {0};
  alignas(8) char src[32] = {0};
  MarkApplicationMemory(dst, sizeof(dst), /*is_copy=*/true);
  MarkApplicationMemory(src, sizeof(src), /*is_copy=*/true);

  memcpy_fn(dst, src, sizeof(dst));
  EXPECT_FALSE(IsMarkedAsCopy(dst, sizeof(dst)));
  EXPECT_TRUE(IsMarkedAsCopy(src, sizeof(src)));
}

TEST_F(CopyProfInterceptorsTest, MemmoveMarksOnlyDestinationAsNonCopy) {
  alignas(8) char buf[32] = {0};
  MarkApplicationMemory(buf, sizeof(buf), /*is_copy=*/true);

  // Overlapping move of `[0, 16)` to `[8, 24)`.
  memmove_fn(buf + 8, buf, /*size=*/16);
  EXPECT_TRUE(IsMarkedAsCopy(buf, /*num_bytes=*/8));
  EXPECT_FALSE(IsMarkedAsCopy(buf + 8, /*num_bytes=*/16));
  EXPECT_TRUE(IsMarkedAsCopy(buf + 24, /*num_bytes=*/8));
}

// Zero-sized writes must not touch shadow memory.
TEST_F(CopyProfInterceptorsTest, ZeroSizedWriteLeavesShadowUnchanged) {
  alignas(8) char buf[16] = {0};
  MarkApplicationMemory(buf, sizeof(buf), /*is_copy=*/true);

  memset_fn(buf + 3, /*c=*/1, /*size=*/0);
  memcpy_fn(buf + 3, buf, /*size=*/0);
  EXPECT_TRUE(IsMarkedAsCopy(buf, sizeof(buf)));
}

// A copy c'tor may copy (trivially copyable) members via `memcpy`, which must
// mark the copy as such.
TEST_F(CopyProfInterceptorsTest, WriteDuringCopyMarksAsCopy) {
  alignas(8) char obj[32] = {0};
  alignas(8) char other[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  memcpy_fn(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));
}

// Like stores, writes by library functions during destruction must not clear
// the copy shadow bits.
TEST_F(CopyProfInterceptorsTest, WriteDuringDtorLeavesShadowUnchanged) {
  alignas(8) char obj[32] = {0};
  alignas(8) char other[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  memset_fn(obj, /*c=*/0, sizeof(obj));
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
}

// Modifying a copy via a library function makes it a necessary copy.
TEST_F(CopyProfInterceptorsTest, ModifyingCopyViaLibraryCallClearsCopy) {
  alignas(8) char obj[32] = {0};
  alignas(8) char other[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  ASSERT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));

  memset_fn(obj + 4, /*c=*/1, /*size=*/1);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_FALSE(__copyprof_state.is_transitive_copy);
}

}  // namespace
}  // namespace __copyprof
