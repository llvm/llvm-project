//===-- sanitizer_posix_test.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for POSIX-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#  include <pthread.h>
#  include <sys/mman.h>

#  include <algorithm>
#  include <numeric>

#  include "gtest/gtest.h"
#  include "sanitizer_common/sanitizer_common.h"

namespace __sanitizer {

static pthread_key_t key;
static bool destructor_executed;

extern "C"
void destructor(void *arg) {
  uptr iter = reinterpret_cast<uptr>(arg);
  if (iter > 1) {
    ASSERT_EQ(0, pthread_setspecific(key, reinterpret_cast<void *>(iter - 1)));
    return;
  }
  destructor_executed = true;
}

extern "C"
void *thread_func(void *arg) {
  return reinterpret_cast<void*>(pthread_setspecific(key, arg));
}

static void SpawnThread(uptr iteration) {
  destructor_executed = false;
  pthread_t tid;
  ASSERT_EQ(0, pthread_create(&tid, 0, &thread_func,
                              reinterpret_cast<void *>(iteration)));
  void *retval;
  ASSERT_EQ(0, pthread_join(tid, &retval));
  ASSERT_EQ(0, retval);
}

TEST(SanitizerCommon, PthreadDestructorIterations) {
  ASSERT_EQ(0, pthread_key_create(&key, &destructor));
  SpawnThread(GetPthreadDestructorIterations());
  EXPECT_TRUE(destructor_executed);
  SpawnThread(GetPthreadDestructorIterations() + 1);
#if SANITIZER_SOLARIS
  // Solaris continues calling destructors beyond PTHREAD_DESTRUCTOR_ITERATIONS.
  EXPECT_TRUE(destructor_executed);
#else
  EXPECT_FALSE(destructor_executed);
#endif
  ASSERT_EQ(0, pthread_key_delete(key));
}

TEST(SanitizerCommon, IsAccessibleMemoryRange) {
  const int page_size = GetPageSize();
  InternalMmapVector<char> buffer(3 * page_size);
  uptr mem = reinterpret_cast<uptr>(buffer.data());
  // Protect the middle page.
  mprotect((void *)(mem + page_size), page_size, PROT_NONE);
  EXPECT_TRUE(IsAccessibleMemoryRange(mem, page_size - 1));
  EXPECT_TRUE(IsAccessibleMemoryRange(mem, page_size));
  EXPECT_FALSE(IsAccessibleMemoryRange(mem, page_size + 1));
  EXPECT_TRUE(IsAccessibleMemoryRange(mem + page_size - 1, 1));
  EXPECT_FALSE(IsAccessibleMemoryRange(mem + page_size - 1, 2));
  EXPECT_FALSE(IsAccessibleMemoryRange(mem + 2 * page_size - 1, 1));
  EXPECT_TRUE(IsAccessibleMemoryRange(mem + 2 * page_size, page_size));
  EXPECT_FALSE(IsAccessibleMemoryRange(mem, 3 * page_size));
  EXPECT_FALSE(IsAccessibleMemoryRange(0x0, 2));
}

TEST(SanitizerCommon, IsAccessibleMemoryRangeLarge) {
  InternalMmapVector<char> buffer(10000 * GetPageSize());
  EXPECT_TRUE(IsAccessibleMemoryRange(reinterpret_cast<uptr>(buffer.data()),
                                      buffer.size()));
}

TEST(SanitizerCommon, TryMemCpy) {
  std::vector<char> src(10000000);
  std::iota(src.begin(), src.end(), 123);
  std::vector<char> dst;

  // Don't use ::testing::ElementsAreArray or similar, as the huge output on an
  // error is not helpful.

  dst.assign(1, 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));

  dst.assign(100, 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));

  dst.assign(534, 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));

  dst.assign(GetPageSize(), 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));

  dst.assign(src.size(), 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));

  dst.assign(src.size() - 1, 0);
  EXPECT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));
}

TEST(SanitizerCommon, TryMemCpyNull) {
  std::vector<char> dst(100);
  EXPECT_FALSE(TryMemCpy(dst.data(), nullptr, dst.size()));
}

TEST(SanitizerCommon, MemCpyAccessible) {
  const int page_num = 1000;
  const int page_size = GetPageSize();
  InternalMmapVector<char> src(page_num * page_size);
  std::iota(src.begin(), src.end(), 123);
  std::vector<char> dst;
  std::vector<char> exp = {src.begin(), src.end()};

  // Protect some pages.
  for (int i = 7; i < page_num; i *= 2) {
    mprotect(src.data() + i * page_size, page_size, PROT_NONE);
    std::fill(exp.data() + i * page_size, exp.data() + (i + 1) * page_size, 0);
  }

  dst.assign(src.size(), 0);
  EXPECT_FALSE(TryMemCpy(dst.data(), src.data(), dst.size()));

  // Full page aligned range with mprotect pages.
  dst.assign(src.size(), 0);
  MemCpyAccessible(dst.data(), src.data(), dst.size());
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), exp.begin()));

  // Misaligned range with mprotect pages.
  size_t offb = 3;
  size_t offe = 7;
  dst.assign(src.size() - offb - offe, 0);
  MemCpyAccessible(dst.data(), src.data() + offb, dst.size());
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), exp.begin() + offb));

  // Misaligned range with ends in mprotect pages.
  offb = 3 + 7 * page_size;
  offe = 7 + 14 * page_size;
  dst.assign(src.size() - offb - offe, 0);
  MemCpyAccessible(dst.data(), src.data() + offb, dst.size());
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), exp.begin() + offb));
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
