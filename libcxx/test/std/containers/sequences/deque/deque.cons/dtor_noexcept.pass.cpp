//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// constexpr since C++26

// ~deque() // implied noexcept;

// UNSUPPORTED: c++03

#include <cassert>
#include <deque>
#include <type_traits>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_alloc {
  typedef T value_type;
  some_alloc(const some_alloc&);
  ~some_alloc() noexcept(false);
  void allocate(std::size_t);
};

#if TEST_STD_VER >= 26
struct NonTrivial {
  int value;

  constexpr NonTrivial(int v = 0) : value(v) {}
  constexpr NonTrivial(const NonTrivial&) = default;
  constexpr ~NonTrivial() {}
};

constexpr bool test() {
  std::deque<NonTrivial> d;
  for (int i = 0; i != 50; ++i)
    d.emplace_back(i);
  assert(d.size() == 50);
  assert(d.front().value == 0);
  assert(d.back().value == 49);
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  test();
  static_assert(test());
#endif

  {
    typedef std::deque<MoveOnly> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
  {
    typedef std::deque<MoveOnly, test_allocator<MoveOnly>> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
  {
    typedef std::deque<MoveOnly, other_allocator<MoveOnly>> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
#if defined(_LIBCPP_VERSION)
  {
    typedef std::deque<MoveOnly, some_alloc<MoveOnly>> C;
    static_assert(!std::is_nothrow_destructible<C>::value, "");
  }
#endif // _LIBCPP_VERSION

  return 0;
}
