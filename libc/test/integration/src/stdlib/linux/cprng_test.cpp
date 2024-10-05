//===-- Unittests for CPRNG -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/stdlib/linux/cprng.h"
#include "test/IntegrationTest/test.h"

namespace LIBC_NAMESPACE_DECL {
namespace cprng {
void smoke_test() {
  auto result = generate<uint32_t>();
  ASSERT_TRUE(result.has_value());
}
void bounded_test() {
  for (uint32_t bound = 1; bound < 5000; ++bound) {
    auto result = generate_bounded_u32(bound);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(*result < bound);
  }
}
void threaded_bounded_test() {
  pthread_t threads[10];
  for (auto &thread : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(
                  &thread, nullptr,
                  [](void *) -> void * {
                    bounded_test();
                    return nullptr;
                  },
                  nullptr),
              0);
  }
  for (auto &thread : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread, nullptr), 0);
  }
}
} // namespace cprng
} // namespace LIBC_NAMESPACE_DECL

TEST_MAIN() {
  LIBC_NAMESPACE::cprng::smoke_test();
  LIBC_NAMESPACE::cprng::bounded_test();
  LIBC_NAMESPACE::cprng::threaded_bounded_test();
  return 0;
}
