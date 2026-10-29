//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for thrd_yield.
///
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"
#include "src/threads/thrd_yield.h"

#include "test/IntegrationTest/test.h"

#include <threads.h>

static int worker(void *) {
  LIBC_NAMESPACE::thrd_yield();
  return 0;
}

void test_yield() {
  LIBC_NAMESPACE::thrd_yield();

  thrd_t thread;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&thread, worker, nullptr),
            static_cast<int>(thrd_success));
  int retval = -1;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(thread, &retval),
            static_cast<int>(thrd_success));
  ASSERT_EQ(retval, 0);
}

TEST_MAIN() {
  test_yield();
  return 0;
}
