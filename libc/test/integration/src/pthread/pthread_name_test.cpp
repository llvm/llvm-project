//===-- Tests for pthread_equal -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getname_np.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_self.h"
#include "src/pthread/pthread_setname_np.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h> // uintptr_t

using string_view = LIBC_NAMESPACE::cpp::string_view;

char child_thread_name_buffer[16];
pthread_mutex_t mutex;

static void *child_func(void *) {
  LIBC_NAMESPACE::pthread_mutex_lock(&mutex);
  auto self = LIBC_NAMESPACE::pthread_self();
  LIBC_NAMESPACE::pthread_getname_np(self, child_thread_name_buffer, 16);
  LIBC_NAMESPACE::pthread_mutex_unlock(&mutex);
  return nullptr;
}

TEST_MAIN() {
  // We init and lock the mutex so that we guarantee that the child thread is
  // waiting after startup.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&mutex, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&mutex), 0);

  auto main_thread = LIBC_NAMESPACE::pthread_self();
  const char MAIN_THREAD_NAME[] = "main_thread";
  char thread_name_buffer[16];
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setname_np(main_thread, MAIN_THREAD_NAME),
            0);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_getname_np(main_thread, thread_name_buffer, 16),
      0);
  ASSERT_EQ(string_view(MAIN_THREAD_NAME),
            string_view(reinterpret_cast<const char *>(thread_name_buffer)));

  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, child_func, nullptr),
            0);
  // This new thread should of course not be equal to the main thread.
  const char CHILD_THREAD_NAME[] = "child_thread";
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setname_np(th, CHILD_THREAD_NAME), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getname_np(th, thread_name_buffer, 16), 0);
  ASSERT_EQ(string_view(CHILD_THREAD_NAME),
            string_view(reinterpret_cast<const char *>(thread_name_buffer)));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&mutex), 0);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
  ASSERT_EQ(uintptr_t(retval), uintptr_t(nullptr));
  // Make sure that the child thread saw it name correctly.
  ASSERT_EQ(
      string_view(CHILD_THREAD_NAME),
      string_view(reinterpret_cast<const char *>(child_thread_name_buffer)));

  LIBC_NAMESPACE::pthread_mutex_destroy(&mutex);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_setname_np(
                main_thread, "a really long name for a thread"),
            ERANGE);
  char smallbuf[1];
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getname_np(main_thread, smallbuf, 1),
            ERANGE);

  return 0;
}
