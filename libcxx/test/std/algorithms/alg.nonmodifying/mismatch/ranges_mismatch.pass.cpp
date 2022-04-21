//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template <input_iterator I1, sentinel_for<_I1> S1, input_iterator I2, sentinel_for<_I2> S2,
//           class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
// constexpr mismatch_result<_I1, _I2>
// ranges::mismatch()(I1 first1, S1 last1, I2 first2, S2 last2, Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {})

// template <input_range R1, input_range R2,
//           class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
// constexpr mismatch_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>>
// ranges::mismatch(R1&& r1, R2&& r2, Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {})

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>

#include "test_iterators.h"

template <class Iter1, class Iter2>
constexpr void test_iterators(Iter1 begin1, Iter1 end1, Iter2 begin2, Iter2 end2, int* expected1, int* expected2) {
  using Expected = std::ranges::mismatch_result<Iter1, Iter2>;
  std::same_as<Expected> auto ret = std::ranges::mismatch(std::move(begin1), sentinel_wrapper<Iter1>(std::move(end1)),
                                                          std::move(begin2), sentinel_wrapper<Iter2>(std::move(end2)));
  assert(base(ret.in1) == expected1);
  assert(base(ret.in2) == expected2);
}

template <class Iter1, class Iter2>
constexpr void test_iters() {
  int a[] = {1, 2, 3, 4, 5};
  int b[] = {1, 2, 3, 5, 4};

  test_iterators(Iter1(a), Iter1(a + 5), Iter2(b), Iter2(b + 5), a + 3, b + 3);
}

constexpr bool test() {
  test_iters<cpp17_input_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, forward_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, random_access_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<cpp17_input_iterator<int*>, int*>();

  test_iters<cpp20_input_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, forward_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, random_access_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<cpp20_input_iterator<int*>, int*>();

  test_iters<forward_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<forward_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<forward_iterator<int*>, forward_iterator<int*>>();
  test_iters<forward_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<forward_iterator<int*>, random_access_iterator<int*>>();
  test_iters<forward_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<forward_iterator<int*>, int*>();

  test_iters<bidirectional_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, forward_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, random_access_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<bidirectional_iterator<int*>, int*>();

  test_iters<random_access_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<random_access_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<random_access_iterator<int*>, forward_iterator<int*>>();
  test_iters<random_access_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<random_access_iterator<int*>, random_access_iterator<int*>>();
  test_iters<random_access_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<random_access_iterator<int*>, int*>();

  test_iters<contiguous_iterator<int*>, cpp17_input_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, cpp20_input_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, forward_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, bidirectional_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, random_access_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, contiguous_iterator<int*>>();
  test_iters<contiguous_iterator<int*>, int*>();

  test_iters<int*, cpp17_input_iterator<int*>>();
  test_iters<int*, cpp20_input_iterator<int*>>();
  test_iters<int*, forward_iterator<int*>>();
  test_iters<int*, bidirectional_iterator<int*>>();
  test_iters<int*, random_access_iterator<int*>>();
  test_iters<int*, contiguous_iterator<int*>>();
  test_iters<int*, int*>();

  { // test with a range
    std::array<int, 5> a = {1, 2, 3, 4, 5};
    std::array<int, 5> b = {1, 2, 3, 5, 4};
    using Expected = std::ranges::mismatch_result<int*, int*>;
    std::same_as<Expected> auto ret = std::ranges::mismatch(a, b);
    assert(ret.in1 == a.begin() + 3);
    assert(ret.in2 == b.begin() + 3);
  }

  { // test with non-iterator sentinel
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {1, 2, 3, 5, 4};

    using Iter = int*;
    using Sentinel = sentinel_wrapper<Iter>;
    using Expected = std::ranges::mismatch_result<Iter, Iter>;

    std::same_as<Expected> auto r = std::ranges::mismatch(Iter(a), Sentinel(a + 5), Iter(b), Sentinel(b + 5));
    assert(r.in1 == a + 3);
    assert(r.in2 == b + 3);
  }

  { // test with different array sizes
    {
      int a[] = {1, 2, 3};
      int b[] = {1, 2};
      test_iterators(a, a + 3, b, b + 2, a + 2, b + 2);
      using Expected = std::ranges::mismatch_result<int*, int*>;
      std::same_as<Expected> auto ret = std::ranges::mismatch(a, b);
      assert(ret.in1 == a + 2);
      assert(ret.in2 == b + 2);
    }
    {
      int a[] = {1, 2};
      int b[] = {1, 2, 3};
      test_iterators(a, a + 2, b, b + 3, a + 2, b + 2);
      using Expected = std::ranges::mismatch_result<int*, int*>;
      std::same_as<Expected> auto ret = std::ranges::mismatch(a, b);
      assert(ret.in1 == a + 2);
      assert(ret.in2 == b + 2);
    }
  }

  { // test with borrowed ranges
    int r1[] = {1, 2, 3, 4, 5};
    int r2[] = {1, 2, 3, 5, 4};

    using Expected = std::ranges::mismatch_result<int*, int*>;
    {
      std::same_as<Expected> auto ret = std::ranges::mismatch(r1, std::views::all(r2));
      assert(ret.in1 == r1 + 3);
      assert(ret.in2 == r2 + 3);
    }
    {
      std::same_as<Expected> auto ret = std::ranges::mismatch(std::views::all(r1), r2);
      assert(ret.in1 == r1 + 3);
      assert(ret.in2 == r2 + 3);
    }
    {
      std::same_as<Expected> auto ret = std::ranges::mismatch(std::views::all(r1), std::views::all(r2));
      assert(ret.in1 == r1 + 3);
      assert(ret.in2 == r2 + 3);
    }
  }

  { // test structured bindings
    int a[] = {1, 2, 3, 4};
    int b[] = {1, 2, 4, 8, 16};
    auto [ai, bi] = std::ranges::mismatch(a, b);
    assert(ai == a + 2);
    assert(bi == b + 2);
    auto [aj, bj] = std::ranges::mismatch(a, a+4, b, b+5);
    assert(aj == a + 2);
    assert(bj == b + 2);
  }

  { // test predicate
    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      int b[] = {6, 5, 8, 2, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a + 8, b, b + 8, std::ranges::greater{});
      assert(ret.in1 == a + 4);
      assert(ret.in2 == b + 4);
      assert(*ret.in1 == 5);
      assert(*ret.in2 == 5);
    }

    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      int b[] = {6, 5, 8, 2, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, b, std::ranges::greater{});
      assert(ret.in1 == a + 4);
      assert(ret.in2 == b + 4);
      assert(*ret.in1 == 5);
      assert(*ret.in2 == 5);
    }
  }

  { // test projection
    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      int b[] = {6, 5, 8, 2, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, b,
                                       std::ranges::greater{},
                                       [](int i) { return i == 5 ? +100 : i; },
                                       [](int i) { return i == 5 ? -100 : i; });
      assert(ret.in1 == a + 5);
      assert(ret.in2 == b + 5);
      assert(*ret.in1 == 1);
      assert(*ret.in2 == 1);
    }
    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a,
                                       std::less<double>{},
                                       [](int i) { return i * 1.01; },
                                       [c = 0](int i) mutable { return c++ < 5 ? i * 1.02 : i; });
      assert(ret.in1 == a + 5);
      assert(ret.in2 == a + 5);
      assert(*ret.in1 == 1);
      assert(*ret.in2 == 1);
    }
    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      int b[] = {6, 5, 8, 2, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a + 8, b, b + 8,
                                       std::ranges::greater{},
                                       [](int i) { return i == 5 ? +100 : i; },
                                       [](int i) { return i == 5 ? -100 : i; });
      assert(ret.in1 == a + 5);
      assert(ret.in2 == b + 5);
      assert(*ret.in1 == 1);
      assert(*ret.in2 == 1);
    }
    {
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a + 8, a, a + 8,
                                       std::less<double>{},
                                       [](int i) { return i * 1.01; },
                                       [c = 0](int i) mutable { return c++ < 5 ? i * 1.02 : i; });
      assert(ret.in1 == a + 5);
      assert(ret.in2 == a + 5);
      assert(*ret.in1 == 1);
      assert(*ret.in2 == 1);
    }
  }

  { // test predicate and projection call count
    {
      int pred_count = 0;
      int proj1_count = 0;
      int proj2_count = 0;
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a,
                                       [&](int lhs, int rhs) { ++pred_count; return lhs == rhs; },
                                       [&](int i) { ++proj1_count; return i; },
                                       [&](int i) { ++proj2_count; return i; });
      assert(ret.in1 == a + 8);
      assert(ret.in2 == a + 8);
      assert(pred_count == 8);
      assert(proj1_count == 8);
      assert(proj2_count == 8);
    }
    {
      int pred_count = 0;
      int proj1_count = 0;
      int proj2_count = 0;
      int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
      auto ret = std::ranges::mismatch(a, a + 8, a, a + 8,
                                       [&](int lhs, int rhs) { ++pred_count; return lhs == rhs; },
                                       [&](int i) { ++proj1_count; return i; },
                                       [&](int i) { ++proj2_count; return i; });
      assert(ret.in1 == a + 8);
      assert(ret.in2 == a + 8);
      assert(pred_count == 8);
      assert(proj1_count == 8);
      assert(proj2_count == 8);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
