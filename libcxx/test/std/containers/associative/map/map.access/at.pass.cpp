//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

//       mapped_type& at(const key_type& k); // constexpr since C++26
// const mapped_type& at(const key_type& k) const; // constexpr since C++26

#include <cassert>
#include <map>
#include <stdexcept>
#include <type_traits>

#include "min_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::pair<const int, double> V;
    V ar[] = {
        V(1, 1.5),
        V(2, 2.5),
        V(3, 3.5),
        V(4, 4.5),
        V(5, 5.5),
        V(7, 7.5),
        V(8, 8.5),
    };
    std::map<int, double> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 7);
    assert(m.at(1) == 1.5);
    m.at(1) = -1.5;
    assert(m.at(1) == -1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
#ifndef TEST_HAS_NO_EXCEPTIONS

// throwing is not allowed in constexpr
#  if TEST_STD_VER < 26
    try {
      TEST_IGNORE_NODISCARD m.at(6);
      assert(false);
    } catch (std::out_of_range&) {
    }
#  endif

#endif
    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
  {
    typedef std::pair<const int, double> V;
    V ar[] = {
        V(1, 1.5),
        V(2, 2.5),
        V(3, 3.5),
        V(4, 4.5),
        V(5, 5.5),
        V(7, 7.5),
        V(8, 8.5),
    };
    const std::map<int, double> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 7);
    assert(m.at(1) == 1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
// throwing is not allowed in constexpr
#if TEST_STD_VER < 26
    try {
      TEST_IGNORE_NODISCARD m.at(6);
      assert(false);
    } catch (std::out_of_range&) {
    }
#endif

    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
#if TEST_STD_VER >= 11
  // #ifdef VINAY_DISABLE_FOR_NOW
  {
    typedef std::pair<const int, double> V;
    V ar[] = {
        V(1, 1.5),
        V(2, 2.5),
        V(3, 3.5),
        V(4, 4.5),
        V(5, 5.5),
        V(7, 7.5),
        V(8, 8.5),
    };

    // std::__tree_node<std::__value_type<int,double>, min_pointer<void>> d;
    // std::__tree_node_base<min_pointer<void>> b  = d;
    using Base = std::__tree_node_base<min_pointer<void>>;

    using Derived = std::__tree_node<std::__value_type<int, double>, min_pointer<void>>;
    static_assert(std::is_base_of_v<Base, Derived>);

    // using BaseP = min_pointer<Base>;
    // using DerivedP = min_pointer<Derived>;
    // static_assert(std::is_base_of_v<BaseP, DerivedP>);
    // DerivedP dp(nullptr);
    // (void)dp;

    // BaseP bp =static_cast<BaseP>(dp);
    // (void)bp;

    std::map<int, double, std::less<int>, min_allocator<V>> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 7);
    assert(m.at(1) == 1.5);
    m.at(1) = -1.5;
    assert(m.at(1) == -1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
#  ifndef TEST_HAS_NO_EXCEPTIONS

// throwing is not allowed in constexpr
#    if TEST_STD_VER < 26
    try {
      TEST_IGNORE_NODISCARD m.at(6);
      assert(false);
    } catch (std::out_of_range&) {
    }
#    endif
#  endif
    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
  {
    typedef std::pair<const int, double> V;
    V ar[] = {
        V(1, 1.5),
        V(2, 2.5),
        V(3, 3.5),
        V(4, 4.5),
        V(5, 5.5),
        V(7, 7.5),
        V(8, 8.5),
    };
    const std::map<int, double, std::less<int>, min_allocator<V>> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    assert(m.size() == 7);
    assert(m.at(1) == 1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
#  ifndef TEST_HAS_NO_EXCEPTIONS
// throwing is not allowed in constexpr
#    if TEST_STD_VER < 26
    try {
      TEST_IGNORE_NODISCARD m.at(6);
      assert(false);
    } catch (std::out_of_range&) {
    }
#    endif
#  endif
    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
#endif
  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
