//===-- Tests for thread detach functionality -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/mutex.h"
#include "src/__support/threads/thread.h"
#include "test/IntegrationTest/test.h"

LIBC_NAMESPACE::Mutex mutex(false, false, false);

int func(void *) {
  mutex.lock();
  mutex.unlock();
  return 0;
}

void detach_simple_test() {
  mutex.lock();
  LIBC_NAMESPACE::Thread th;
  th.run(func, nullptr, nullptr, 0);

  // Since |mutex| is held by the current thread, we guarantee that
  // th is running and hence it is safe to detach. Since the thread is
  // still running, it should be simple detach.
  ASSERT_EQ(th.detach(), int(LIBC_NAMESPACE::DetachType::SIMPLE));

  // We will release |mutex| now to let the thread finish an cleanup itself.
  mutex.unlock();
}

void detach_cleanup_test() {
  mutex.lock();
  LIBC_NAMESPACE::Thread th;
  ASSERT_EQ(0, th.run(func, nullptr));

  // Since |mutex| is held by the current thread, we will release it
  // to let |th| run.
  mutex.unlock();

  // We will wait for |th| to finish. Since it is a joinable thread,
  // we can wait on it safely.
  th.wait();

  // Since |th| is now finished, detaching should cleanup the thread
  // resources.
  ASSERT_EQ(th.detach(), int(LIBC_NAMESPACE::DetachType::CLEANUP));
}

TEST_MAIN() {
  detach_simple_test();
  detach_cleanup_test();
  return 0;
}
