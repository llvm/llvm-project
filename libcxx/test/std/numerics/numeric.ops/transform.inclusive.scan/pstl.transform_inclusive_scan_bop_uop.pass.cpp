//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class ExecutionPolicy,
//           class ForwardIterator1,
//           class ForwardIterator2,
//           class BinaryOperation,
//           class UnaryOperation,
//           class T>
//   ForwardIterator2 transform_inclusive_scan(ExecutionPolicy&& exec,
//                                             ForwardIterator1 first,
//                                             ForwardIterator1 last,
//                                             ForwardIterator2 result,
//                                             BinaryOperation reduce,
//                                             UnaryOperation transform);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(transform_inclusive_scan);

static_assert(sfinae_test_transform_inclusive_scan<int, int*, int*, int*, int (*)(int, int), int (*)(int)>);
static_assert(!sfinae_test_transform_inclusive_scan< std::execution::parallel_policy,
                                                     int*,
                                                     int*,
                                                     int*,
                                                     int (*)(int, int),
                                                     int (*)(int)>);

struct add_one {
  template <typename T>
  constexpr T operator()(T x) const {
    return x + 1;
  }
};

template <typename T>
constexpr auto triangle(T n) {
  return n * (n + 1) / 2;
}

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]  = {0};
      auto res = std::transform_inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<int>{}, [](int x) {
            return x;
          });
      static_assert(std::is_same_v<decltype(res), Iter2>);
    }
    { // Empty
      int a[] = {-1};
      std::transform_inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(a)), std::plus<>{}, add_one{});
      assert(a[0] == -1);
    }
    { // Scan a uniform array
      int a[10];
      std::fill(std::begin(a), std::end(a), 3);
      std::transform_inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == (static_cast<int>(i) + 1) * 4);
    }
    { // Scan an increasing array starting from 0
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      std::transform_inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(triangle(i) + i + 1));
    }
    { // Scan an increasing array starting from 1
      int a[10];
      std::iota(std::begin(a), std::end(a), 1);
      std::transform_inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(triangle(i + 1) + i + 1));
    }
    { // Test against pre-computed expected results
      auto test = [&](auto first, auto last, auto reduce, auto transform, auto exp_first, auto exp_last) {
        int out[5];
        // Not in place
        auto end = std::transform_inclusive_scan(policy, first, last, Iter2(std::begin(out)), reduce, transform);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
        // In place
        std::copy(first, last, out);
        end = std::transform_inclusive_scan(
            policy, Iter2(std::begin(out)), end, Iter2(std::begin(out)), reduce, transform);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
      };
      const int a[]              = {1, 3, 5, 7, 9};
      const int exp_plus_inc_0[] = {2, 6, 12, 20, 30};          // plus, add_one, init = 0
      const int exp_mul_inc_0[]  = {2, 8, 48, 384, 3840};       // multiplies, add_one, init = 0
      const int exp_plus_neg_0[] = {-1, -4, -9, -16, -25};      // plus, negate, init = 0
      const int exp_mul_neg_0[]  = {-1, 3, -15, 105, -945};     // multiplies, negate, init = 0
      static_assert(std::size(a) == std::size(exp_plus_inc_0)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_inc_0));  // just to be sure
      static_assert(std::size(a) == std::size(exp_plus_neg_0)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_neg_0));  // just to be sure
      std::plus<> plus;
      std::multiplies<> mult;
      std::negate<> negate;
      for (unsigned int i = 0; i < std::size(a); ++i) {
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), plus, add_one{}, exp_plus_inc_0, exp_plus_inc_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), mult, add_one{}, exp_mul_inc_0, exp_mul_inc_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), plus, negate, exp_plus_neg_0, exp_plus_neg_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), mult, negate, exp_mul_neg_0, exp_mul_neg_0 + i);
      }
    }
    { // Constuct iota from std::plus<> and add_one
      int ia[1073];
      int ib[1073];
      std::fill(std::begin(ia), std::end(ia), 0);
      runway_sample(std::size(ia) + 1, [&](size_t i) {
        Iter2 r = std::transform_inclusive_scan(
            policy, Iter1(std::begin(ia)), Iter1(std::begin(ia) + i), Iter2(std::begin(ib)), std::plus<>{}, add_one{});
        assert(r == Iter2(std::begin(ib) + i));
        for (size_t j = 0; j < i; ++j) {
          assert(ib[j] == static_cast<int>(j) + 1);
        }
        std::fill(std::begin(ib), std::begin(ib) + i, -1);
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});
  return 0;
}
