//===-- Unittests for Expected --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/type_traits.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::expected;
using LIBC_NAMESPACE::cpp::unexpected;

class MoveOnly {
  int *destroyed;

public:
  explicit MoveOnly(int *destroyed) : destroyed(destroyed) {}

  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;

  MoveOnly(MoveOnly &&other) : destroyed(other.destroyed) {
    other.destroyed = nullptr;
  }

  ~MoveOnly() {
    if (destroyed)
      ++*destroyed;
  }
};

static_assert(
    LIBC_NAMESPACE::cpp::is_trivially_destructible_v<expected<int, int>>,
    "expected should be trivially destructible for trivial payloads");
static_assert(
    !LIBC_NAMESPACE::cpp::is_trivially_destructible_v<expected<MoveOnly, int>>,
    "expected should be nontrivially destructible for nontrivial "
    "value payloads");
static_assert(
    !LIBC_NAMESPACE::cpp::is_trivially_destructible_v<expected<int, MoveOnly>>,
    "expected should be nontrivially destructible for nontrivial "
    "error payloads");

TEST(LlvmLibcExpectedTest, DestroysNonTrivialValue) {
  int destroyed = 0;
  {
    expected<MoveOnly, int> value{MoveOnly(&destroyed)};
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(static_cast<bool>(value));
  }
  ASSERT_EQ(destroyed, 1);
}

TEST(LlvmLibcExpectedTest, DestroysNonTrivialError) {
  int destroyed = 0;
  {
    expected<int, MoveOnly> error{unexpected<MoveOnly>{MoveOnly(&destroyed)}};
    ASSERT_FALSE(error.has_value());
    ASSERT_FALSE(static_cast<bool>(error));
  }
  ASSERT_EQ(destroyed, 1);
}
