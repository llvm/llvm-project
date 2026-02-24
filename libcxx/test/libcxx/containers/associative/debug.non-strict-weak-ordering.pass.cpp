//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>
// <map>

// This test ensures that libc++ detects when std::set or std::map are used with a
// predicate that is not a strict weak ordering when the debug mode is enabled.

// REQUIRES: libcpp-hardening-mode=debug
// UNSUPPORTED: c++03, c++11, c++14

#include <map>
#include <set>
#include <utility>

#include "check_assertion.h"
#include "test_macros.h"

struct InvalidLess {
  bool operator()(int a, int b) const {
    if (b == 0)
      return true;
    return static_cast<bool>(a % b);
  }
};

int main() {
  // std::set
  {
    // find
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.find(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).find(4), "Comparator does not induce a strict weak ordering");
    }

    // upper_bound
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.upper_bound(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).upper_bound(4), "Comparator does not induce a strict weak ordering");
    }

    // lower_bound
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.lower_bound(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).lower_bound(4), "Comparator does not induce a strict weak ordering");
    }

    // equal_range
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.equal_range(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).equal_range(4), "Comparator does not induce a strict weak ordering");
    }

    // count
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.count(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).count(4), "Comparator does not induce a strict weak ordering");
    }

    // contains
#if TEST_STD_VER >= 20
    {
      std::set<int, InvalidLess> s = {1, 2, 3, 4};
      TEST_LIBCPP_ASSERT_FAILURE(s.contains(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).contains(4), "Comparator does not induce a strict weak ordering");
    }
#endif
  }

  // std::map
  {
    using X   = int;
    X const x = 99;

    // find
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.find(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).find(4), "Comparator does not induce a strict weak ordering");
    }

    // upper_bound
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.upper_bound(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).upper_bound(4), "Comparator does not induce a strict weak ordering");
    }

    // lower_bound
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.lower_bound(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).lower_bound(4), "Comparator does not induce a strict weak ordering");
    }

    // equal_range
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.equal_range(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).equal_range(4), "Comparator does not induce a strict weak ordering");
    }

    // count
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.count(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).count(4), "Comparator does not induce a strict weak ordering");
    }

    // contains
#if TEST_STD_VER >= 20
    {
      std::map<int, X, InvalidLess> s = {{1, x}, {2, x}, {3, x}, {4, x}};
      TEST_LIBCPP_ASSERT_FAILURE(s.contains(4), "Comparator does not induce a strict weak ordering");
      TEST_LIBCPP_ASSERT_FAILURE(std::as_const(s).contains(4), "Comparator does not induce a strict weak ordering");
    }
#endif
  }
}
