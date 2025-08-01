//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// size_type max_size() const noexcept;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_allocator.h"
#include "test_macros.h"

constexpr bool test() {
  {
    using A1 = limited_allocator<int, 10>;
    using A2 = limited_allocator<int, 20>;
    using C  = std::flat_multimap<int, int, std::less<int>, std::vector<int, A1>, std::vector<int, A2>>;
    ASSERT_SAME_TYPE(C::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(C::size_type, std::size_t);
    const C c;
    ASSERT_NOEXCEPT(c.max_size());
    ASSERT_SAME_TYPE(decltype(c.max_size()), C::size_type);
    assert(c.max_size() <= 10);
    LIBCPP_ASSERT(c.max_size() == 10);
  }
  {
    using A1 = limited_allocator<int, 10>;
    using A2 = limited_allocator<int, 20>;
    using C  = std::flat_multimap<int, int, std::less<int>, std::vector<int, A2>, std::vector<int, A1>>;
    ASSERT_SAME_TYPE(C::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(C::size_type, std::size_t);
    const C c;
    ASSERT_NOEXCEPT(c.max_size());
    ASSERT_SAME_TYPE(decltype(c.max_size()), C::size_type);
    assert(c.max_size() <= 10);
    LIBCPP_ASSERT(c.max_size() == 10);
  }
  {
    using A = limited_allocator<int, (size_t)-1>;
    using C = std::flat_multimap<int, int, std::less<int>, std::vector<int, A>, std::vector<int, A>>;
    ASSERT_SAME_TYPE(C::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(C::size_type, std::size_t);
    const C::size_type max_dist = static_cast<C::size_type>(std::numeric_limits<C::difference_type>::max());
    const C c;
    ASSERT_NOEXCEPT(c.max_size());
    ASSERT_SAME_TYPE(decltype(c.max_size()), C::size_type);
    assert(c.max_size() <= max_dist);
    LIBCPP_ASSERT(c.max_size() == max_dist);
  }
  {
    typedef std::flat_multimap<char, char> C;
    ASSERT_SAME_TYPE(C::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(C::size_type, std::size_t);
    const C::size_type max_dist = static_cast<C::size_type>(std::numeric_limits<C::difference_type>::max());
    const C c;
    ASSERT_NOEXCEPT(c.max_size());
    ASSERT_SAME_TYPE(decltype(c.max_size()), C::size_type);
    assert(c.max_size() <= max_dist);
    assert(c.max_size() <= alloc_max_size(std::allocator<char>()));
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
