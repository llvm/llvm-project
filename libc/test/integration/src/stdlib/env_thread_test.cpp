//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Multi-threaded integration test for environment variable operations.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdlib/getenv.h"
#include "src/stdlib/setenv.h"
#include "src/stdlib/unsetenv.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"
#include "test/IntegrationTest/test.h"

#include <threads.h>

static LIBC_NAMESPACE::cpp::Atomic<bool> stop_flag(false);

static int writer_thread_a(void *) {
  for (int i = 0; i < 500; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("CONCURRENT_VAR_A", "value_a1", 1), 0);
    const char *val = LIBC_NAMESPACE::getenv("CONCURRENT_VAR_A");
    ASSERT_NE(val, nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("CONCURRENT_VAR_A", "value_a2", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::unsetenv("CONCURRENT_VAR_A"), 0);
  }
  return 0;
}

static int writer_thread_b(void *) {
  for (int i = 0; i < 500; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("CONCURRENT_VAR_B", "value_b1", 1), 0);
    const char *val = LIBC_NAMESPACE::getenv("CONCURRENT_VAR_B");
    ASSERT_NE(val, nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("CONCURRENT_VAR_B", "value_b2", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::unsetenv("CONCURRENT_VAR_B"), 0);
  }
  return 0;
}

static int reader_thread(void *) {
  while (!stop_flag.load(LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED)) {
    const char *val_a = LIBC_NAMESPACE::getenv("CONCURRENT_VAR_A");
    if (val_a) {
      LIBC_NAMESPACE::cpp::string_view sv(val_a);
      ASSERT_TRUE(sv == "value_a1" || sv == "value_a2");
    }

    const char *val_b = LIBC_NAMESPACE::getenv("CONCURRENT_VAR_B");
    if (val_b) {
      LIBC_NAMESPACE::cpp::string_view sv(val_b);
      ASSERT_TRUE(sv == "value_b1" || sv == "value_b2");
    }
  }
  return 0;
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  thrd_t w1, w2, r1, r2;

  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&r1, reader_thread, nullptr),
            static_cast<int>(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&r2, reader_thread, nullptr),
            static_cast<int>(thrd_success));

  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&w1, writer_thread_a, nullptr),
            static_cast<int>(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&w2, writer_thread_b, nullptr),
            static_cast<int>(thrd_success));

  int res = 0;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(w1, &res),
            static_cast<int>(thrd_success));
  ASSERT_EQ(res, 0);

  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(w2, &res),
            static_cast<int>(thrd_success));
  ASSERT_EQ(res, 0);

  stop_flag.store(true, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);

  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(r1, &res),
            static_cast<int>(thrd_success));
  ASSERT_EQ(res, 0);

  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(r2, &res),
            static_cast<int>(thrd_success));
  ASSERT_EQ(res, 0);

  return 0;
}
