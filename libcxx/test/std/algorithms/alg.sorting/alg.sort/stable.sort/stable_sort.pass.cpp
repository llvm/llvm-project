//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class RandomAccessIterator>
//     constexpr void               // constexpr in C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last);

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=20000000

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <random>
#include <vector>

#include "count_new.h"
#include "test_macros.h"

template <class RI>
void
test_sort_helper(RI f, RI l)
{
    typedef typename std::iterator_traits<RI>::value_type value_type;
    typedef typename std::iterator_traits<RI>::difference_type difference_type;

    if (f != l)
    {
        difference_type len = l - f;
        value_type* save(new value_type[len]);
        do
        {
            std::copy(f, l, save);
            std::stable_sort(save, save+len);
            assert(std::is_sorted(save, save+len));
        } while (std::next_permutation(f, l));
        delete [] save;
    }
}

template <class RI>
void
test_sort_driver_driver(RI f, RI l, int start, RI real_last)
{
    for (RI i = l; i > f + start;)
    {
        *--i = start;
        if (f == i)
        {
            test_sort_helper(f, real_last);
        }
    if (start > 0)
        test_sort_driver_driver(f, i, start-1, real_last);
    }
}

template <class RI>
void
test_sort_driver(RI f, RI l, int start)
{
    test_sort_driver_driver(f, l, start, l);
}

template <int sa>
void
test_sort_()
{
    int ia[sa];
    for (int i = 0; i < sa; ++i)
    {
        test_sort_driver(ia, ia+sa, i);
    }
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> init_saw_tooth_pattern() {
  std::array<int, N> array;
  for (int i = 0, x = 0; i < N; ++i) {
    array[i] = x;
    if (++x == M)
      x = 0;
  }
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> sort_saw_tooth_pattern() {
  std::array<int, N> array = init_saw_tooth_pattern<N, M>();
  std::stable_sort(array.begin(), array.end());
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> sort_already_sorted() {
  std::array<int, N> array = sort_saw_tooth_pattern<N, M>();
  std::stable_sort(array.begin(), array.end());
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> sort_reversely_sorted() {
  std::array<int, N> array = sort_saw_tooth_pattern<N, M>();
  std::reverse(array.begin(), array.end());
  std::stable_sort(array.begin(), array.end());
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> sort_swapped_sorted_ranges() {
  std::array<int, N> array = sort_saw_tooth_pattern<N, M>();
  std::swap_ranges(array.begin(), array.begin() + N / 2, array.begin() + N / 2);
  std::stable_sort(array.begin(), array.end());
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 std::array<int, N> sort_reversely_swapped_sorted_ranges() {
  std::array<int, N> array = sort_saw_tooth_pattern<N, M>();
  std::reverse(array.begin(), array.end());
  std::swap_ranges(array.begin(), array.begin() + N / 2, array.begin() + N / 2);
  std::stable_sort(array.begin(), array.end());
  return array;
}

template <int N, int M>
TEST_CONSTEXPR_CXX26 void test_larger_sorts() {
  static_assert(N > 0, "");
  static_assert(M > 0, "");

  { // test saw tooth pattern
    std::array<int, N> array = sort_saw_tooth_pattern<N, M>();
    assert(std::is_sorted(array.begin(), array.end()));
  }

#if TEST_STD_VER >= 26
  if !consteval
#endif
  { // test random pattern
    // random-number generators not constexpr-friendly
    static std::mt19937 randomness;
    std::array<int, N> array = init_saw_tooth_pattern<N, M>();
    std::shuffle(array.begin(), array.end(), randomness);
    std::stable_sort(array.begin(), array.end());
    assert(std::is_sorted(array.begin(), array.end()));
  }

  { // test sorted pattern
    std::array<int, N> array = sort_already_sorted<N, M>();
    assert(std::is_sorted(array.begin(), array.end()));
  }

  { // test reverse sorted pattern
    std::array<int, N> array = sort_reversely_sorted<N, M>();
    assert(std::is_sorted(array.begin(), array.end()));
  }

  { // test swap ranges 2 pattern
    std::array<int, N> array = sort_swapped_sorted_ranges<N, M>();
    assert(std::is_sorted(array.begin(), array.end()));
  }

  { // test reverse swap ranges 2 pattern
    std::array<int, N> array = sort_reversely_swapped_sorted_ranges<N, M>();
    assert(std::is_sorted(array.begin(), array.end()));
  }
}

template <int N>
TEST_CONSTEXPR_CXX26 void test_larger_sorts() {
  test_larger_sorts<N, 1>();
  test_larger_sorts<N, 2>();
  test_larger_sorts<N, 3>();
  test_larger_sorts<N, N / 2 - 1>();
  test_larger_sorts<N, N / 2>();
  test_larger_sorts<N, N / 2 + 1>();
  test_larger_sorts<N, N - 2>();
  test_larger_sorts<N, N - 1>();
  test_larger_sorts<N, N>();
}

int main(int, char**) {
  { // test null range
    int d = 0;
    std::stable_sort(&d, &d);
#if TEST_STD_VER >= 26
    static_assert((std::stable_sort(&d, &d), true));
#endif
  }

  { // exhaustively test all possibilities up to length 8
    test_sort_<1>();
    test_sort_<2>();
    test_sort_<3>();
    test_sort_<4>();
    test_sort_<5>();
    test_sort_<6>();
    test_sort_<7>();
    test_sort_<8>();
  }

  { // larger sorts
    // run- and compile-time tests
    test_larger_sorts<256>();
    test_larger_sorts<257>();
#if TEST_STD_VER >= 26
    static_assert((test_larger_sorts<256>(), true));
    static_assert((test_larger_sorts<257>(), true));
#endif

    // only runtime tests bc. error: "constexpr evaluation hit maximum step limit"
    test_larger_sorts<499>();
    test_larger_sorts<500>();
    test_larger_sorts<997>();
    test_larger_sorts<1000>();
    test_larger_sorts<1009>();
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // check that the algorithm works without memory
    std::vector<int> vec(150, 3);
    getGlobalMemCounter()->throw_after = 0;
    std::stable_sort(vec.begin(), vec.end());
  }
#endif

  return 0;
}
