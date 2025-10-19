//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// ~map() // implied noexcept; constexpr since C++26

// UNSUPPORTED: c++03

#include <map>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_comp {
  typedef T value_type;
  ~some_comp() noexcept(false);
  bool operator()(const T&, const T&) const noexcept { return false; }
};

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::pair<const MoveOnly, MoveOnly> V;
  {
    typedef std::map<MoveOnly, MoveOnly> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
  {
    typedef std::map<MoveOnly, MoveOnly, std::less<MoveOnly>, test_allocator<V>> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
  {
    typedef std::map<MoveOnly, MoveOnly, std::less<MoveOnly>, other_allocator<V>> C;
    static_assert(std::is_nothrow_destructible<C>::value, "");
  }
#if defined(_LIBCPP_VERSION)
  {
    typedef std::map<MoveOnly, MoveOnly, some_comp<MoveOnly>> C;
    static_assert(!std::is_nothrow_destructible<C>::value, "");
  }
#endif // _LIBCPP_VERSION
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
