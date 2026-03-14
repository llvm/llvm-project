//===-- Tests for TSS API like tss_set, tss_get etc. ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "src/threads/thrd_exit.h"
#include "src/threads/thrd_join.h"
#include "src/threads/tss_create.h"
#include "src/threads/tss_delete.h"
#include "src/threads/tss_get.h"
#include "src/threads/tss_set.h"
#include "test/IntegrationTest/test.h"

#include <threads.h>

static constexpr int THREAD_DATA_INITVAL = 0x1234;
static constexpr int THREAD_DATA_FINIVAL = 0x4321;
static constexpr int THREAD_RUN_VAL = 0x600D;

int child_thread_data = THREAD_DATA_INITVAL;
int main_thread_data = THREAD_DATA_INITVAL;

tss_t key;
void dtor(void *data) {
  auto *v = reinterpret_cast<int *>(data);
  *v = THREAD_DATA_FINIVAL;
}

int func(void *obj) {
  ASSERT_EQ(LIBC_NAMESPACE::tss_set(key, &child_thread_data), thrd_success);
  int *d = reinterpret_cast<int *>(LIBC_NAMESPACE::tss_get(key));
  ASSERT_TRUE(d != nullptr);
  ASSERT_EQ(&child_thread_data, d);
  ASSERT_EQ(*d, THREAD_DATA_INITVAL);
  *reinterpret_cast<int *>(obj) = THREAD_RUN_VAL;
  return 0;
}

TEST_MAIN() {
  ASSERT_EQ(LIBC_NAMESPACE::tss_create(&key, &dtor), thrd_success);
  ASSERT_EQ(LIBC_NAMESPACE::tss_set(key, &main_thread_data), thrd_success);
  int *d = reinterpret_cast<int *>(LIBC_NAMESPACE::tss_get(key));
  ASSERT_TRUE(d != nullptr);
  ASSERT_EQ(&main_thread_data, d);
  ASSERT_EQ(*d, THREAD_DATA_INITVAL);

  thrd_t th;
  int arg = 0xBAD;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&th, &func, &arg), thrd_success);
  int retval = THREAD_DATA_INITVAL; // Init to some non-zero val.
  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(th, &retval), thrd_success);
  ASSERT_EQ(retval, 0);
  ASSERT_EQ(arg, THREAD_RUN_VAL);
  ASSERT_EQ(child_thread_data, THREAD_DATA_FINIVAL);

  LIBC_NAMESPACE::tss_delete(key);

  return 0;
}
