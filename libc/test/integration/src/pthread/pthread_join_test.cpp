//===-- Tests for pthread_join-- ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"

#include "src/__support/libc_errno.h"

#include "test/IntegrationTest/test.h"
#include <pthread.h>

static void *simpleFunc(void *) { return nullptr; }
static void nullJoinTest() {
  pthread_t Tid;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&Tid, nullptr, simpleFunc, nullptr),
            0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(Tid, nullptr), 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_MAIN() {
  libc_errno = 0;
  nullJoinTest();
  return 0;
}
