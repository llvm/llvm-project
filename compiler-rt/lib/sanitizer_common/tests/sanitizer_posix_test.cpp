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

TEST(SanitizerCommon, TryMemCpyProtected) {
  const int page_size = GetPageSize();
  InternalMmapVector<char> src(3 * page_size);
  std::iota(src.begin(), src.end(), 123);
  std::vector<char> dst;
  // Protect the middle page.
  mprotect(src.data() + page_size, page_size, PROT_NONE);

  dst.assign(src.size(), 0);
  EXPECT_FALSE(TryMemCpy(dst.data(), src.data(), dst.size()));

  mprotect(src.data() + page_size, page_size, PROT_READ | PROT_WRITE);
  EXPECT_TRUE(std::equal(dst.begin(), dst.end(), src.begin()));
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
