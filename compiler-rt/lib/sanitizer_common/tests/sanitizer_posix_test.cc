//===-- sanitizer_posix_test.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for POSIX-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "sanitizer_common/sanitizer_common.h"
#include "gtest/gtest.h"

#include <pthread.h>
#include <sys/mman.h>

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
  EXPECT_FALSE(destructor_executed);
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
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
