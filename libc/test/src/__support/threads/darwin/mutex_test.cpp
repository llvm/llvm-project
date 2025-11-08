//===-- Unittests for Darwin's Mutex ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/darwin/mutex.h"
#include "src/__support/threads/mutex_common.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSupportThreadsMutexTest, SmokeTest) {
  LIBC_NAMESPACE::Mutex mutex;
  ASSERT_EQ(mutex.lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.try_lock(), LIBC_NAMESPACE::MutexError::BUSY);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::NONE);
  ASSERT_EQ(mutex.unlock(), LIBC_NAMESPACE::MutexError::UNLOCK_WITHOUT_LOCK);
}

// TEST(LlvmLibcSupportThreadsRawMutexTest, Timeout) {
//   LIBC_NAMESPACE::RawMutex mutex;
//   ASSERT_TRUE(mutex.lock());
//   timespec ts;
//   LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &ts);
//   ts.tv_sec += 1;
//   // Timeout will be respected when deadlock happens.
//   auto timeout = LIBC_NAMESPACE::internal::AbsTimeout::from_timespec(ts,
//   false); ASSERT_TRUE(timeout.has_value());
//   // The following will timeout
//   ASSERT_FALSE(mutex.lock(*timeout));
//   ASSERT_TRUE(mutex.unlock());
//   // Test that the mutex works after the timeout.
//   ASSERT_TRUE(mutex.lock());
//   ASSERT_TRUE(mutex.unlock());
//   // If a lock can be acquired directly, expired timeout will not count.
//   // Notice that the timeout is already reached during preivous deadlock.
//   ASSERT_TRUE(mutex.lock(*timeout));
//   ASSERT_TRUE(mutex.unlock());
// }
//
// TEST(LlvmLibcSupportThreadsRawMutexTest, PSharedLock) {
//   struct SharedData {
//     LIBC_NAMESPACE::RawMutex mutex;
//     LIBC_NAMESPACE::cpp::Atomic<size_t> finished;
//     int data;
//   };
//   void *addr =
//       LIBC_NAMESPACE::mmap(nullptr, sizeof(SharedData), PROT_READ |
//       PROT_WRITE,
//                            MAP_ANONYMOUS | MAP_SHARED, -1, 0);
//   ASSERT_NE(addr, MAP_FAILED);
//   auto *shared = reinterpret_cast<SharedData *>(addr);
//   shared->data = 0;
//   LIBC_NAMESPACE::RawMutex::init(&shared->mutex);
//   // Avoid pull in our own implementation of pthread_t.
// #ifdef SYS_fork
//   long pid = LIBC_NAMESPACE::syscall_impl<long>(SYS_fork);
// #elif defined(SYS_clone)
//   long pid = LIBC_NAMESPACE::syscall_impl<long>(SYS_clone, SIGCHLD, 0);
// #endif
//   for (int i = 0; i < 10000; ++i) {
//     shared->mutex.lock(LIBC_NAMESPACE::cpp::nullopt, true);
//     shared->data++;
//     shared->mutex.unlock(true);
//   }
//   // Mark the thread as finished.
//   shared->finished.fetch_add(1);
//   // let the child exit early to avoid output pollution
//   if (pid == 0)
//     LIBC_NAMESPACE::exit(0);
//   while (shared->finished.load() != 2)
//     LIBC_NAMESPACE::sleep_briefly();
//   ASSERT_EQ(shared->data, 20000);
//   LIBC_NAMESPACE::munmap(addr, sizeof(SharedData));
// }
