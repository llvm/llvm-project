//===-- copyprof_allocator_test.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_allocator.h"

#include <errno.h>
#include <pthread.h>
#include <string.h>

#include "copyprof/copyprof_interface_internal.h"
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_allocator.h"

namespace __copyprof {
namespace {

class CopyProfAllocatorTest : public testing::Test {
 protected:
  void SetUp() override { __copyprof_init_once(); }
};

TEST_F(CopyProfAllocatorTest, ZeroSizedAllocation) {
  void* zero_buf = CopyProfAllocate(/*size=*/0, /*align=*/8);
  ASSERT_NE(zero_buf, nullptr);
  CopyProfDeallocate(zero_buf, /*size=*/0, /*align=*/8);
}

TEST_F(CopyProfAllocatorTest, ThreadLocalAllocatorCacheAndDrainOnExit) {
  auto worker = +[](void* /*arg*/) -> void* {
    void* local_buf = CopyProfAllocate(/*size=*/64, /*align=*/8);
    CopyProfDeallocate(local_buf, /*size=*/64, /*align=*/8);
    return CopyProfAllocate(/*size=*/128, /*align=*/8);
  };
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, worker, /*arg=*/nullptr),
            0);
  void* cross_thread_buf = nullptr;
  ASSERT_EQ(pthread_join(thread, &cross_thread_buf), 0);
  ASSERT_NE(cross_thread_buf, nullptr);
  CopyProfDeallocate(cross_thread_buf, /*size=*/128, /*align=*/8);
}

// The exiting thread may still (de)allocate after its cache has been committed
// back, e.g. in TSD destructors that run later.
TEST_F(CopyProfAllocatorTest, AllocationAfterThreadCacheCommitBackSucceeds) {
  static pthread_key_t key;
  static void* late_buf;
  late_buf = nullptr;
  auto destructor = +[](void* value) {
    if (value == &key) {
      // Invoking `pthread_setspecific` forces `pthread_exit` to perform another
      // pass over TSD destructors, resulting in execution of the code path
      // below afterwards.
      pthread_setspecific(key, &late_buf);
      return;
    }
    late_buf = CopyProfMalloc(/*size=*/16);
    CopyProfFree(late_buf);
  };
  ASSERT_EQ(pthread_key_create(&key, destructor), 0);
  auto worker = +[](void* /*arg*/) -> void* {
    // Allocate once to create the thread cache and arm its TSD destructor.
    CopyProfFree(CopyProfMalloc(/*size=*/16));
    pthread_setspecific(key, &key);
    return nullptr;
  };
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, worker, /*arg=*/nullptr),
            0);
  ASSERT_EQ(pthread_join(thread, /*thread_return=*/nullptr), 0);
  pthread_key_delete(key);
  EXPECT_NE(late_buf, nullptr);
}

TEST_F(CopyProfAllocatorTest, CallocReturnsZeroedMemory) {
  constexpr uptr kSize = 32;
  // Dirty a chunk and free it again, s.t. `CopyProfCalloc` likely reuses it.
  void* dirty_buf = CopyProfMalloc(kSize);
  ASSERT_NE(dirty_buf, nullptr);
  memset(dirty_buf, /*c=*/0xab, kSize);
  CopyProfFree(dirty_buf);

  char* buf = static_cast<char*>(CopyProfCalloc(/*num=*/1, kSize));
  ASSERT_NE(buf, nullptr);
  for (uptr i = 0; i < kSize; ++i) EXPECT_EQ(buf[i], 0) << "at offset " << i;
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, CallocOverflowReturnsNull) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  errno = 0;

  // `num * size` overflows.
  void* buf = CopyProfCalloc(/*num=*/2, /*size=*/~uptr{0});
  const int calloc_errno = errno;
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(calloc_errno, ENOMEM);
}

TEST_F(CopyProfAllocatorTest, ReallocPreservesContents) {
  constexpr uptr kSize = 16;
  char* buf = static_cast<char*>(CopyProfMalloc(kSize));
  ASSERT_NE(buf, nullptr);
  for (uptr i = 0; i < kSize; ++i) buf[i] = static_cast<char>(i);

  char* resized_buf =
      static_cast<char*>(CopyProfRealloc(buf, /*size=*/2 * kSize));
  ASSERT_NE(resized_buf, nullptr);
  for (uptr i = 0; i < kSize; ++i)
    EXPECT_EQ(resized_buf[i], static_cast<char>(i)) << "at offset " << i;
  CopyProfFree(resized_buf);
}

TEST_F(CopyProfAllocatorTest, ReallocOfNullptrAllocates) {
  void* buf = CopyProfRealloc(/*ptr=*/nullptr, /*size=*/32);
  EXPECT_NE(buf, nullptr);
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, ReallocToZeroSizeReturnsNull) {
  void* buf = CopyProfMalloc(/*size=*/32);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfRealloc(buf, /*size=*/0), nullptr);
}

TEST_F(CopyProfAllocatorTest, AllocateWithInvalidAlignmentSetsEinval) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);

  for (uptr align : {uptr{0}, uptr{3}}) {
    errno = 0;
    void* buf = CopyProfAllocate(/*size=*/32, align);
    EXPECT_EQ(buf, nullptr) << "for align " << align;
    EXPECT_EQ(errno, EINVAL) << "for align " << align;
  }

  SetAllocatorMayReturnNull(may_return_null);
}

TEST_F(CopyProfAllocatorTest, AllocateReturnsMemoryWithRequestedAlignment) {
  constexpr uptr kAlignment = 256;
  void* buf = CopyProfAllocate(/*size=*/16, kAlignment);
  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), kAlignment));
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest,
       PosixMemalignReturnsMemoryWithRequestedAlignment) {
  constexpr uptr kAlignment = 64;
  void* buf = nullptr;
  ASSERT_EQ(CopyProfPosixMemalign(&buf, kAlignment, /*size=*/100), 0);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), kAlignment));
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, PosixMemalignWithInvalidAlignmentReturnsEinval) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  void* buf = nullptr;

  // Not a multiple of `sizeof(void*)`.
  const int result = CopyProfPosixMemalign(&buf, /*alignment=*/4, /*size=*/16);
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(result, EINVAL);
  EXPECT_EQ(buf, nullptr);
}

TEST_F(CopyProfAllocatorTest, VallocReturnsPageAlignedMemory) {
  void* buf = CopyProfValloc(/*size=*/16);
  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), GetPageSizeCached()));
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, PvallocRoundsSizeUpToPageSize) {
  void* buf = CopyProfPvalloc(/*size=*/1);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), GetPageSizeCached());
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, ReallocarrayAllocatesNumTimesSize) {
  void* buf = CopyProfReallocarray(/*ptr=*/nullptr, /*num=*/4, /*size=*/8);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), 32u);
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, ReallocarrayOverflowReturnsNull) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  errno = 0;

  // `num * size` overflows.
  void* buf =
      CopyProfReallocarray(/*ptr=*/nullptr, /*num=*/2, /*size=*/~uptr{0});
  const int reallocarray_errno = errno;
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(reallocarray_errno, ENOMEM);
}

TEST_F(CopyProfAllocatorTest, MallocUsableSizeReturnsRequestedSize) {
  void* buf = CopyProfMalloc(/*size=*/10);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), 10u);
  CopyProfFree(buf);
}

TEST_F(CopyProfAllocatorTest, MallocUsableSizeOfUnownedMemoryIsZero) {
  char stack_buf[16] = {0};
  EXPECT_EQ(CopyProfMallocUsableSize(stack_buf), 0u);
}

}  // namespace
}  // namespace __copyprof
