//===-- Test of Lambda Locks ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/threads/lambda_lock.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "test/IntegrationTest/test.h"

void simple_addition() {
  LIBC_NAMESPACE::LambdaLock<int> sum{0};
  pthread_t threads[10];
  for (int i = 0; i < 10; ++i) {
    LIBC_NAMESPACE::pthread_create(
        &threads[i], nullptr,
        [](void *arg) -> void * {
          auto *lock = static_cast<LIBC_NAMESPACE::LambdaLock<int> *>(arg);
          for (int j = 0; j < 1000; ++j)
            lock->enqueue([j](int &sum) { sum += j; });
          return nullptr;
        },
        &sum);
  }
  for (int i = 0; i < 10; ++i)
    LIBC_NAMESPACE::pthread_join(threads[i], nullptr);

  ASSERT_EQ(sum.get_unsafe(), 4995000);
}

template <size_t LIMIT> void string_concat() {
  static char buffer[10001];
  LIBC_NAMESPACE::LambdaLock<LIBC_NAMESPACE::cpp::StringStream> shared{buffer};
  pthread_t threads[10];
  for (int i = 0; i < 10; ++i) {
    LIBC_NAMESPACE::pthread_create(
        &threads[i], nullptr,
        [](void *arg) -> void * {
          auto *lock = static_cast<decltype(shared) *>(arg);
          for (int j = 0; j < 100; ++j)
            for (int c = 0; c < 10; ++c)
              lock->enqueue(
                  [c](LIBC_NAMESPACE::cpp::StringStream &data) { data << c; },
                  LIMIT);
          return nullptr;
        },
        &shared);
  }
  for (int i = 0; i < 10; ++i)
    LIBC_NAMESPACE::pthread_join(threads[i], nullptr);

  LIBC_NAMESPACE::cpp::string_view x = shared.get_unsafe().str();
  ASSERT_EQ(x.size(), 10000);
  int count[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (char c : x)
    count[c - '0']++;

  for (int i = 0; i < 10; ++i)
    ASSERT_EQ(count[i], 1000);
}

TEST_MAIN() {
  simple_addition();
  string_concat<0>();
  string_concat<1>();
  string_concat<2>();
  string_concat<17>();
  string_concat<LIBC_NAMESPACE::cpp::numeric_limits<size_t>::max()>();
  return 0;
}
