//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// __non_propagating_cache& operator=(__non_propagating_cache&&);

// ADDITIONAL_COMPILE_FLAGS: -Wno-self-assign

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

template<bool NoexceptMove>
struct MoveAssignable {
  int x;
  constexpr explicit MoveAssignable(int i) : x(i) { }
  MoveAssignable(MoveAssignable&&) = default;
  constexpr MoveAssignable& operator=(MoveAssignable&& other) noexcept(NoexceptMove) {
    x = other.x;
    other.x = -1;
    return *this;
  }
};

struct NotMoveAssignable {
  int x;
  constexpr explicit NotMoveAssignable(int i) : x(i) { }
  NotMoveAssignable(NotMoveAssignable&&) = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&) = delete;
};

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;
  static_assert(std::is_nothrow_move_assignable_v<Cache>);

  // Assign to an empty cache
  {
    Cache a; a.__emplace(3);
    Cache b;

    Cache& result = (b = std::move(a));
    assert(&result == &b);
    assert(!b.__has_value()); // make sure we don't propagate
    assert(!a.__has_value()); // make sure we disengage the source
  }

  // Assign to a non-empty cache
  {
    Cache a; a.__emplace(3);
    Cache b; b.__emplace(5);

    Cache& result = (b = std::move(a));
    assert(&result == &b);
    assert(!b.__has_value()); // make sure we don't propagate
    assert(!a.__has_value()); // make sure we disengage the source
  }

  // Self-assignment should clear the cache (case with empty cache)
  {
    Cache b;

    Cache& result = (b = std::move(b));
    assert(&result == &b);
    assert(!b.__has_value());
  }

  // Self-assignment should clear the cache (case with non-empty cache)
  {
    Cache b; b.__emplace(5);

    Cache& result = (b = std::move(b));
    assert(&result == &b);
    assert(!b.__has_value());
  }
}

constexpr bool tests() {
  test<MoveAssignable<true>>();
  test<MoveAssignable<false>>();
  test<NotMoveAssignable>();
  test<int>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
