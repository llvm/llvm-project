//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class ...Args>
// constexpr T& __emplace(Args&& ...);

#include <ranges>

#include <cassert>
#include <tuple>

template<int I>
struct X {
  int value = -1;
  template<int J>
  friend constexpr bool operator==(X const& a, X<J> const& b) { return I == J && a.value == b.value; }
};

struct NonMovable {
  int value = -1;
  NonMovable() = default;
  constexpr explicit NonMovable(int v) : value(v) { }
  NonMovable(NonMovable&&) = delete;
  NonMovable& operator=(NonMovable&&) = delete;
};

constexpr bool test() {
  {
    using T = std::tuple<>;
    using Cache = std::ranges::__non_propagating_cache<T>;
    Cache cache;
    T& result = cache.__emplace();
    assert(&result == &*cache);
    assert(result == T());
  }

  {
    using T = std::tuple<X<0>>;
    using Cache = std::ranges::__non_propagating_cache<T>;
    Cache cache;
    T& result = cache.__emplace();
    assert(&result == &*cache);
    assert(result == T());
  }
  {
    using T = std::tuple<X<0>>;
    using Cache = std::ranges::__non_propagating_cache<T>;
    Cache cache;
    T& result = cache.__emplace(X<0>{0});
    assert(&result == &*cache);
    assert(result == T(X<0>{0}));
  }

  {
    using T = std::tuple<X<0>, X<1>>;
    using Cache = std::ranges::__non_propagating_cache<T>;
    Cache cache;
    T& result = cache.__emplace();
    assert(&result == &*cache);
    assert(result == T());
  }
  {
    using T = std::tuple<X<0>, X<1>>;
    using Cache = std::ranges::__non_propagating_cache<T>;
    Cache cache;
    T& result = cache.__emplace(X<0>{0}, X<1>{1});
    assert(&result == &*cache);
    assert(result == T(X<0>{0}, X<1>{1}));
  }

  // Make sure that we do not require the type to be movable when we emplace it.
  // Move elision should be performed instead, see http://eel.is/c++draft/range.nonprop.cache#note-1.
  {
    using Cache = std::ranges::__non_propagating_cache<NonMovable>;
    Cache cache;
    NonMovable& result = cache.__emplace();
    assert(&result == &*cache);
    assert(result.value == -1);
  }

  return true;
}

int main(int, char**) {
  static_assert(test());
  test();
  return 0;
}
