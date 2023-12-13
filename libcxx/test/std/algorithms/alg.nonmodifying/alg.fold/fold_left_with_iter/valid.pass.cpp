//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<input_iterator I, sentinel_for<I> S, class T,
//          indirectly-binary-left-foldable<T, I> F>
//   constexpr see below ranges::fold_left_with_iter(I first, S last, T init, F f);
//
// template<input_range R, class T, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   constexpr see below ranges::fold_left_with_iter(R&& r, T init, F f);

#include <algorithm>
#include <cassert>
#include <vector>
#include <functional>

#include "test_range.h"
#include "../gaussian_sum.h"

constexpr bool test() {
  {
    auto data   = std::vector<int>{1, 2, 3, 4};
    auto result = std::ranges::fold_left_with_iter(data.begin(), data.begin(), 0, std::plus());

    assert(result.in == data.begin());
    assert(result.result == 0);

    auto range   = std::span(data.data(), 0);
    auto rresult = std::ranges::fold_left_with_iter(range, 0, std::plus());

    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }
  {
    auto data           = std::vector<int>{1, 2, 3, 4};
    constexpr auto init = 100.1;
    auto result         = std::ranges::fold_left_with_iter(data.begin(), data.begin(), init, std::plus());

    assert(result.in == data.begin());
    assert(result.result == init);

    auto range   = std::span(data.data(), 0);
    auto rresult = std::ranges::fold_left_with_iter(range, init, std::plus());

    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }
  {
    auto data   = std::vector{1, 3, 5, 7, 9};
    auto result = std::ranges::fold_left_with_iter(data.begin(), data.end(), 0, std::plus());

    assert(result.in == data.end());
    assert(result.result == gaussian_sum(data)); // sum of n ascending odd numbers = n^2

    auto rresult = std::ranges::fold_left_with_iter(data, 0, std::plus());
    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }
  {
    auto data         = std::vector{2, 4, 6, 8, 10, 12};
    auto const result = std::ranges::fold_left_with_iter(data.begin(), data.end(), 0L, std::plus<long>());

    assert(result.in == data.end());
    assert(result.result == gaussian_sum(data));

    auto rresult = std::ranges::fold_left_with_iter(data, 0, std::plus());
    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }
  {
    auto data   = std::vector{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6};
    auto plus   = [](int const x, double const y) { return x + y; };
    auto result = std::ranges::fold_left_with_iter(data.begin(), data.end(), 0.0, plus);

    assert(result.in == data.end());
    assert(result.result == -21.6); // int(  0.0) + -1.1 =   0 + -1.1 =  -1.1
                                    // int(- 1.1) + -2.2 = - 1 + -2.2 =  -3.2
                                    // int(- 3.2) + -3.3 = - 3 + -3.3 =  -6.3
                                    // int(- 6.3) + -4.4 = - 6 + -4.4 = -10.4
                                    // int(-10.4) + -5.5 = -10 + -5.5 = -15.5
                                    // int(-15.5) + -6.6 = -15 + -6.6 = -21.6.

    auto rresult = std::ranges::fold_left_with_iter(data, 0, plus);
    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }
  {
    auto data           = std::vector<double>{1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1};
    auto plus           = [](double const x, double const y) { return static_cast<short>(x + y); };
    constexpr auto init = 10.0;
    auto result         = std::ranges::fold_left_with_iter(data.begin(), data.end(), init, plus);

    assert(result.in == data.end());
    assert(result.result == static_cast<short>(gaussian_sum(data) + init));

    auto rresult = std::ranges::fold_left_with_iter(data, init, plus);
    assert(rresult.in == result.in);
    assert(rresult.result == result.result);
  }

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
