//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   constexpr Iter    // constexpr after c++17
//   lower_bound(Iter first, Iter last, const T& value);

#include <algorithm>
#include <vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17

TEST_CONSTEXPR bool eq(int a, int b) { return a == b; }

TEST_CONSTEXPR bool test_constexpr() {
    int ia[] = {1, 3, 6, 7};

    return (std::lower_bound(std::begin(ia), std::end(ia), 2) == ia+1)
        && (std::lower_bound(std::begin(ia), std::end(ia), 3) == ia+1)
        && (std::lower_bound(std::begin(ia), std::end(ia), 9) == std::end(ia))
        ;
    }
#endif


template <class Iter, class T>
void
test(Iter first, Iter last, const T& value)
{
  std::size_t strides{};
  std::size_t displacement{};
  stride_counting_iterator f(first, &strides, &displacement);
  stride_counting_iterator l(last, &strides, &displacement);

  auto i = std::lower_bound(f, l, value);
  for (auto j = f; j != i; ++j)
    assert(*j < value);
  for (auto j = i; j != l; ++j)
    assert(!(*j < value));

  auto len = std::distance(first, last);
  assert(strides <= 2.5 * len + 1);
  assert(displacement <= 2.5 * len + 1);
}

template <class Iter>
void
test()
{
    const unsigned N = 1000;
    const int M = 10;
    std::vector<int> v(N);
    int x = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        v[i] = x;
        if (++x == M)
            x = 0;
    }
    std::sort(v.begin(), v.end());
    for (x = 0; x <= M; ++x)
        test(Iter(v.data()), Iter(v.data()+v.size()), x);
}

int main(int, char**)
{
    int d[] = {0, 1, 2, 3};
    for (int* e = d; e <= d+4; ++e)
        for (int x = -1; x <= 4; ++x)
            test(d, e, x);

    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();

#if TEST_STD_VER > 17
    static_assert(test_constexpr());
#endif

  return 0;
}
