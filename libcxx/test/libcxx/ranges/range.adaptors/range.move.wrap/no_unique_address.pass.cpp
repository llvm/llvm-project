//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// clang-cl and cl currently don't support [[no_unique_address]]
// XFAIL: msvc

#include <ranges>

#include <cassert>
#include <utility>

#include "test_macros.h"

template <class T, bool ExpectNoUniqueAddress>
void test_no_unique_address() {
  struct Test {
    [[no_unique_address]] std::ranges::__movable_box<T> box_;
    bool b2;
  };

  if constexpr (ExpectNoUniqueAddress) {
    static_assert(sizeof(Test) == sizeof(bool));
  } else {
    static_assert(sizeof(Test) > sizeof(bool));
  }
}

struct Copyable {};

struct NotCopyAssignable {
  constexpr NotCopyAssignable()                          = default;
  constexpr NotCopyAssignable(const NotCopyAssignable&)  = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
};

struct NotMoveAssignable {
  constexpr NotMoveAssignable()                          = default;
  constexpr NotMoveAssignable(const NotMoveAssignable&)  = default;
  NotMoveAssignable& operator=(const NotMoveAssignable&) = default;
  constexpr NotMoveAssignable(NotMoveAssignable&&)       = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&)      = delete;
};

struct MoveOnly {
  constexpr MoveOnly()                 = default;
  constexpr MoveOnly(const MoveOnly&)  = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  constexpr MoveOnly(MoveOnly&&)       = default;
  MoveOnly& operator=(MoveOnly&&)      = default;
};

struct MoveOnlyNotAssignable {
  constexpr MoveOnlyNotAssignable()                              = default;
  constexpr MoveOnlyNotAssignable(const MoveOnlyNotAssignable&)  = delete;
  MoveOnlyNotAssignable& operator=(const MoveOnlyNotAssignable&) = delete;
  constexpr MoveOnlyNotAssignable(MoveOnlyNotAssignable&&)       = default;
  MoveOnlyNotAssignable& operator=(MoveOnlyNotAssignable&&)      = delete;
};

int main(int, char**) {
  test_no_unique_address<Copyable, true>();
  test_no_unique_address<NotCopyAssignable, false>();
  test_no_unique_address<NotMoveAssignable, false>();

#if TEST_STD_VER >= 23
  test_no_unique_address<MoveOnly, true>();
  test_no_unique_address<MoveOnlyNotAssignable, false>();
#endif

  return 0;
}
