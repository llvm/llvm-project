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

// __non_propagating_cache(__non_propagating_cache const&);

#include <ranges>

#include <cassert>
#include <type_traits>

template<bool NoexceptCopy>
struct CopyConstructible {
  int x;
  constexpr explicit CopyConstructible(int i) : x(i) { }
  constexpr CopyConstructible(CopyConstructible const& other) noexcept(NoexceptCopy) : x(other.x) { }
  CopyConstructible& operator=(CopyConstructible const&) = default;
  constexpr bool operator==(CopyConstructible const& other) const { return x == other.x; }
};

struct NotCopyConstructible {
  int x;
  constexpr explicit NotCopyConstructible(int i) : x(i) { }
  NotCopyConstructible(NotCopyConstructible const&) = delete;
  NotCopyConstructible(NotCopyConstructible&&) = default;
  constexpr bool operator==(NotCopyConstructible const& other) const { return x == other.x; }
};

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;
  static_assert(std::is_nothrow_copy_constructible_v<Cache>);
  Cache a;
  a.__set(T{3});

  // Test with direct initialization
  {
    Cache b(a);
    assert(!b.__has_value()); // make sure we don't propagate

    assert(a.__has_value()); // make sure we don't "steal" from the source
    assert(*a == T{3});      //
  }

  // Test with copy initialization
  {
    Cache b = a;
    assert(!b.__has_value()); // make sure we don't propagate

    assert(a.__has_value()); // make sure we don't "steal" from the source
    assert(*a == T{3});      //
  }
}

constexpr bool tests() {
  test<CopyConstructible<true>>();
  test<CopyConstructible<false>>();
  test<NotCopyConstructible>();
  test<int>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
