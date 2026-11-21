//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    vector2(InputIterator, InputIterator, Allocator = Allocator())
//    -> vector2<typename iterator_traits<InputIterator>::value_type, Allocator>;
//
// template<ranges::input_range R, class Allocator = allocator<ranges::range_value_t<R>>>
//   vector2(from_range_t, R&&, Allocator = Allocator())
//     -> vector2<ranges::range_value_t<R>, Allocator>; // C++23

#include <algorithm>
#include <array>
#include <cassert>
#include <climits> // INT_MAX
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "deduction_guides_sfinae_checks.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

TEST_CONSTEXPR_CXX20 bool tests() {
  //  Test the explicit deduction guides
  {
    const int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector2 vec(std::begin(arr), std::end(arr));

    static_assert(std::is_same_v<decltype(vec), std::vector2<int>>, "");
    assert(std::equal(vec.begin(), vec.end(), std::begin(arr), std::end(arr)));
  }

  {
    const long arr[] = {INT_MAX, 1L, 2L, 3L};
    std::vector2 vec(std::begin(arr), std::end(arr), std::allocator<long>());
    static_assert(std::is_same_v<decltype(vec)::value_type, long>, "");
    assert(vec.size() == 4);
    assert(vec[0] == INT_MAX);
    assert(vec[1] == 1L);
    assert(vec[2] == 2L);
  }

  //  Test the implicit deduction guides

  {
    //  We don't expect this one to work.
    //  std::vector2 vec(std::allocator<int>()); // vector2 (allocator &)
  }

  {
    std::vector2 vec(1, A{}); // vector2 (size_type, T)
    static_assert(std::is_same_v<decltype(vec)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, std::allocator<A>>, "");
    assert(vec.size() == 1);
  }

  {
    std::vector2 vec(1, A{}, test_allocator<A>()); // vector2 (size_type, T, allocator)
    static_assert(std::is_same_v<decltype(vec)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, test_allocator<A>>, "");
    assert(vec.size() == 1);
  }

  {
    std::vector2 vec{1U, 2U, 3U, 4U, 5U}; // vector2(initializer-list)
    static_assert(std::is_same_v<decltype(vec)::value_type, unsigned>, "");
    assert(vec.size() == 5);
    assert(vec[2] == 3U);
  }

  {
    std::vector2 vec({1.0, 2.0, 3.0, 4.0}, test_allocator<double>()); // vector2(initializer-list, allocator)
    static_assert(std::is_same_v<decltype(vec)::value_type, double>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, test_allocator<double>>, "");
    assert(vec.size() == 4);
    assert(vec[3] == 4.0);
  }

  {
    std::vector2<long double> source;
    std::vector2 vec(source); // vector2(vector2 &)
    static_assert(std::is_same_v<decltype(vec)::value_type, long double>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, std::allocator<long double>>, "");
    assert(vec.size() == 0);
  }

#if TEST_STD_VER >= 23
  {
    {
      std::vector2 c(std::from_range, std::array<int, 0>());
      static_assert(std::is_same_v<decltype(c), std::vector2<int>>);
    }

    {
      using Alloc = test_allocator<int>;
      std::vector2 c(std::from_range, std::array<int, 0>(), Alloc());
      static_assert(std::is_same_v<decltype(c), std::vector2<int, Alloc>>);
    }
  }
#endif

  //  A couple of vector2<bool> tests, too!
  {
    std::vector2 vec(3, true); // vector2(initializer-list)
    static_assert(std::is_same_v<decltype(vec)::value_type, bool>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, std::allocator<bool>>, "");
    assert(vec.size() == 3);
    assert(vec[0] && vec[1] && vec[2]);
  }

  {
    std::vector2<bool> source;
    std::vector2 vec(source); // vector2(vector2 &)
    static_assert(std::is_same_v<decltype(vec)::value_type, bool>, "");
    static_assert(std::is_same_v<decltype(vec)::allocator_type, std::allocator<bool>>, "");
    assert(vec.size() == 0);
  }

  {
    typedef test_allocator<short> Alloc;
    typedef test_allocator<int> ConvertibleToAlloc;

    {
      std::vector2<short, Alloc> source;
      std::vector2 vec(source, Alloc(2));
      static_assert(std::is_same_v<decltype(vec), decltype(source)>);
    }

    {
      std::vector2<short, Alloc> source;
      std::vector2 vec(source, ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(vec), decltype(source)>);
    }

    {
      std::vector2<short, Alloc> source;
      std::vector2 vec(std::move(source), Alloc(2));
      static_assert(std::is_same_v<decltype(vec), decltype(source)>);
    }

    {
      std::vector2<short, Alloc> source;
      std::vector2 vec(std::move(source), ConvertibleToAlloc(2));
      static_assert(std::is_same_v<decltype(vec), decltype(source)>);
    }
  }

  SequenceContainerDeductionGuidesSfinaeAway<std::vector2, std::vector2<int>>();

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
