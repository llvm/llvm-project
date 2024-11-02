//===-- Test handling of thread local data --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/thread.h"
#include "utils/IntegrationTest/test.h"

static constexpr int INIT_VAL = 100;
static constexpr int UPDATE_VAL = 123;

static thread_local int tlval = INIT_VAL;

// This function updates the tlval and returns its old value.
int func(void *) {
  int old_tlval = tlval;
  tlval = UPDATE_VAL;
  return old_tlval;
}

void thread_local_test() {
  int retval;

  __llvm_libc::Thread th1;
  th1.run(func, nullptr, nullptr, 0);
  th1.join(&retval);
  ASSERT_EQ(retval, INIT_VAL);

  __llvm_libc::Thread th2;
  th2.run(func, nullptr, nullptr, 0);
  th2.join(&retval);
  ASSERT_EQ(retval, INIT_VAL);
}

TEST_MAIN() {
  // From the main thread, we will update the main thread's tlval.
  // This should not affect the child thread's tlval;
  ASSERT_EQ(tlval, INIT_VAL);
  tlval = UPDATE_VAL;
  ASSERT_EQ(tlval, UPDATE_VAL);
  thread_local_test();
  return 0;
}
