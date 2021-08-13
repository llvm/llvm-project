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

// __non_propagating_cache& operator=(__non_propagating_cache const&);

// ADDITIONAL_COMPILE_FLAGS: -Wno-self-assign

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

template<bool NoexceptCopy>
struct CopyAssignable {
  int x;
  constexpr explicit CopyAssignable(int i) : x(i) { }
  CopyAssignable(CopyAssignable const&) = default;
  constexpr CopyAssignable& operator=(CopyAssignable const& other) noexcept(NoexceptCopy) {
    x = other.x;
    return *this;
  }
  constexpr bool operator==(CopyAssignable const& other) const { return x == other.x; }
};

struct NotCopyAssignable {
  int x;
  constexpr explicit NotCopyAssignable(int i) : x(i) { }
  NotCopyAssignable(NotCopyAssignable const&) = default;
  NotCopyAssignable& operator=(NotCopyAssignable const&) = delete;
  constexpr bool operator==(NotCopyAssignable const& other) const { return x == other.x; }
};

template <class T>
constexpr void test() {
  using Cache = std::ranges::__non_propagating_cache<T>;
  static_assert(std::is_nothrow_copy_assignable_v<Cache>);

  // Assign to an empty cache
  {
    Cache a; a.__set(T{3});
    Cache b;

    Cache& result = (b = a);
    assert(&result == &b);
    assert(!b.__has_value()); // make sure we don't propagate

    assert(a.__has_value()); // make sure we don't "steal" from the source
    assert(*a == T{3});      //
  }

  // Assign to a non-empty cache
  {
    Cache a; a.__set(T{3});
    Cache b; b.__set(T{5});

    Cache& result = (b = a);
    assert(&result == &b);
    assert(!b.__has_value()); // make sure we don't propagate

    assert(a.__has_value()); // make sure we don't "steal" from the source
    assert(*a == T{3});      //
  }

  // Self-assignment should not do anything (case with empty cache)
  {
    Cache b;
    Cache& result = (b = b);
    assert(&result == &b);
    assert(!b.__has_value());
  }

  // Self-assignment should not do anything (case with non-empty cache)
  {
    Cache b; b.__set(T{5});
    Cache& result = (b = b);
    assert(&result == &b);
    assert(b.__has_value());
    assert(*b == T{5});
  }
}

constexpr bool tests() {
  test<CopyAssignable<true>>();
  test<CopyAssignable<false>>();
  test<NotCopyAssignable>();
  test<int>();
  return true;
}

int main(int, char**) {
  static_assert(tests());
  tests();
  return 0;
}
