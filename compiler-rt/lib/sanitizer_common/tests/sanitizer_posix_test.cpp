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

#  include "gmock/gmock.h"
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
  uptr mem = (uptr)mmap(0, 3 * page_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
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

  munmap((void *)mem, 3 * page_size);
}

TEST(SanitizerCommon, IsAccessibleMemoryRangeLarge) {
  const int size = GetPageSize() * 10000;

  uptr mem = (uptr)mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
                        -1, 0);

  EXPECT_TRUE(IsAccessibleMemoryRange(mem, size));

  munmap((void *)mem, size);
}

TEST(SanitizerCommon, TryMemCpy) {
  std::vector<char> src(10000000);
  std::iota(src.begin(), src.end(), 123);
  std::vector<char> dst;

  using ::testing::ElementsAreArray;

  dst.assign(1, 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  dst.assign(100, 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  dst.assign(534, 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  dst.assign(GetPageSize(), 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  dst.assign(src.size(), 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  dst.assign(src.size() - 1, 0);
  ASSERT_TRUE(TryMemCpy(dst.data(), src.data(), dst.size()));
  EXPECT_THAT(dst, ElementsAreArray(src.data(), dst.size()));

  EXPECT_FALSE(TryMemCpy(dst.data(), nullptr, dst.size()));
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
