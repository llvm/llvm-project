//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// reference at(size_type n); // constexpr since C++20

#include <cassert>
#include <memory>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#  include <stdexcept>
#endif

template <typename Allocator>
TEST_CONSTEXPR_CXX20 void test() {
  using C         = std::vector<bool, Allocator>;
  using reference = typename C::reference;
  bool a[]        = {1, 0, 1, 0, 1};
  C v(a, a + sizeof(a) / sizeof(a[0]));
  ASSERT_SAME_TYPE(reference, decltype(v.at(0)));
  assert(v.at(0) == true);
  assert(v.at(1) == false);
  assert(v.at(2) == true);
  assert(v.at(3) == false);
  assert(v.at(4) == true);
  v.at(1) = 1;
  assert(v.at(1) == true);
  v.at(3) = 1;
  assert(v.at(3) == true);
}

template <typename Allocator>
void test_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    bool a[] = {1, 0, 1, 1};
    using C  = std::vector<bool, Allocator>;
    C v(a, a + sizeof(a) / sizeof(a[0]));

    try {
      TEST_IGNORE_NODISCARD v.at(4);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }

    try {
      TEST_IGNORE_NODISCARD v.at(5);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }

    try {
      TEST_IGNORE_NODISCARD v.at(6);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }

    try {
      using size_type = typename C::size_type;
      TEST_IGNORE_NODISCARD v.at(static_cast<size_type>(-1));
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }

  {
    std::vector<bool, Allocator> v;
    try {
      TEST_IGNORE_NODISCARD v.at(0);
      assert(false);
    } catch (std::out_of_range const&) {
      // pass
    } catch (...) {
      assert(false);
    }
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test<std::allocator<bool> >();
  test<min_allocator<bool> >();
  test<test_allocator<bool> >();
  return true;
}

void test_exceptions() {
  test_exception<std::allocator<bool> >();
  test_exception<min_allocator<bool> >();
  test_exception<test_allocator<bool> >();
}

int main(int, char**) {
  tests();
  test_exceptions();

#if TEST_STD_VER >= 20
  static_assert(tests());
#endif

  return 0;
}
