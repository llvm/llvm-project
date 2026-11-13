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
//           class T,
//           class BinaryOperation,
//           class UnaryOperation>
//   ForwardIterator2 transform_exclusive_scan(ExecutionPolicy&& exec,
//                                             ForwardIterator1 first,
//                                             ForwardIterator1 last,
//                                             ForwardIterator2 result,
//                                             T init,
//                                             BinaryOperation reduce,
//                                             UnaryOperation transform);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(transform_exclusive_scan);

static_assert(sfinae_test_transform_exclusive_scan<int, int*, int*, int*, int, int (*)(int, int), int (*)(int)>);
static_assert(!sfinae_test_transform_exclusive_scan<
              std::execution::parallel_policy,
              int*,
              int*,
              int*,
              int,
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
      auto res = std::transform_exclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), 0, std::plus<int>{}, [](int x) {
            return x;
          });
      static_assert(std::is_same_v<decltype(res), Iter2>);
    }
    { // Empty
      int a[] = {-1};
      std::transform_exclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(a)), 10, std::plus<>{}, add_one{});
      assert(a[0] == -1);
    }
    { // Scan a uniform array
      int a[10];
      std::fill(std::begin(a), std::end(a), 3);
      std::transform_exclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), 50, std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == 50 + static_cast<int>(i) * 4);
    }
    { // Scan an increasing array starting from 0
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      std::transform_exclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), 30, std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(30 + triangle(i)));
    }
    { // Scan an increasing array starting from 1
      int a[10];
      std::iota(std::begin(a), std::end(a), 1);
      std::transform_exclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), 40, std::plus<>{}, add_one{});
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(40 + triangle(i) + i));
    }
    { // Test against pre-computed expected results
      auto test = [&](auto first, auto last, auto init, auto reduce, auto transform, auto exp_first, auto exp_last) {
        int out[5];
        // Not in place
        auto end = std::transform_exclusive_scan(policy, first, last, Iter2(std::begin(out)), init, reduce, transform);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
        // In place
        std::copy(first, last, out);
        end = std::transform_exclusive_scan(
            policy, Iter2(std::begin(out)), end, Iter2(std::begin(out)), init, reduce, transform);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
      };
      const int a[]              = {1, 3, 5, 7, 9};
      const int exp_plus_inc_0[] = {0, 2, 6, 12, 20};           // plus, add_one, init = 0
      const int exp_mul_inc_0[]  = {0, 0, 0, 0, 0};             // multiplies, add_one, init = 0
      const int exp_plus_neg_0[] = {0, -1, -4, -9, -16};        // plus, negate, init = 0
      const int exp_mul_neg_0[]  = {0, 0, 0, 0, 0};             // multiplies, negate, init = 0
      const int exp_plus_inc_2[] = {2, 4, 8, 14, 22};           // plus, add_one, init = 2
      const int exp_mul_inc_2[]  = {2, 4, 16, 96, 768};         // multiplies, add_one, init = 2
      const int exp_plus_neg_2[] = {2, 1, -2, -7, -14};         // plus, negate, init = 2
      const int exp_mul_neg_2[]  = {2, -2, 6, -30, 210};        // multiplies, negate, init = 2
      static_assert(std::size(a) == std::size(exp_plus_inc_0)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_inc_0));  // just to be sure
      static_assert(std::size(a) == std::size(exp_plus_neg_0)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_neg_0));  // just to be sure
      static_assert(std::size(a) == std::size(exp_plus_inc_2)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_inc_2));  // just to be sure
      static_assert(std::size(a) == std::size(exp_plus_neg_2)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_neg_2));  // just to be sure
      std::plus<> plus;
      std::multiplies<> mult;
      std::negate<> negate;
      for (unsigned int i = 0; i < std::size(a); ++i) {
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 0, plus, add_one{}, exp_plus_inc_0, exp_plus_inc_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 0, mult, add_one{}, exp_mul_inc_0, exp_mul_inc_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 0, plus, negate, exp_plus_neg_0, exp_plus_neg_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 0, mult, negate, exp_mul_neg_0, exp_mul_neg_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 2, plus, add_one{}, exp_plus_inc_2, exp_plus_inc_2 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 2, mult, add_one{}, exp_mul_inc_2, exp_mul_inc_2 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 2, plus, negate, exp_plus_neg_2, exp_plus_neg_2 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), 2, mult, negate, exp_mul_neg_2, exp_mul_neg_2 + i);
      }
    }
    { // Construct iota from std::plus<> and add_one
      int ia[1073];
      int ib[1073];
      std::fill(std::begin(ia), std::end(ia), 0);
      runway_sample(std::size(ia) + 1, [&](std::size_t i) {
        Iter2 r = std::transform_exclusive_scan(
            policy,
            Iter1(std::begin(ia)),
            Iter1(std::begin(ia) + i),
            Iter2(std::begin(ib)),
            1,
            std::plus<>{},
            add_one{});
        assert(r == Iter2(std::begin(ib) + i));
        for (std::size_t j = 0; j < i; ++j) {
          assert(ib[j] == static_cast<int>(j) + 1);
        }
        std::fill(std::begin(ib), std::begin(ib) + i, -1);
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestUnsignedCharToSize_t {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    //  Make sure that the calculations are done using the init typedef
    unsigned char a[1073];
    std::size_t b[1073];
    std::fill(std::begin(a), std::end(a), static_cast<unsigned char>(254));
    runway_sample(std::size(a) + 1, [&](std::size_t i) {
      std::transform_exclusive_scan(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::begin(a) + i),
          Iter2(std::begin(b)),
          std::size_t{0},
          std::plus<>{},
          add_one{});
      for (std::size_t k = 0; k < i; ++k)
        assert(b[k] == 255 * k);
    });
  }
};

template <class Iter1, class Iter2>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      auto binary_op   = [](const std::string& a, const std::string& b) { return a + b; };
      auto unary_op    = [](const std::string& s) { return s + "!"; };
      std::string init = "0";
      for (std::size_t size : {0, 1, 2, 3, 10, 100, 1000}) {
        std::vector<std::string> data(size);
        for (std::size_t i = 0; i < size; ++i)
          data[i] = std::string(1, static_cast<char>('a' + i % 26));

        std::vector<std::string> expected(size);
        std::transform_exclusive_scan(data.begin(), data.end(), expected.begin(), init, binary_op, unary_op);

        // Not in place
        std::vector<std::string> result(size);
        auto ret = std::transform_exclusive_scan(
            policy,
            Iter1(data.data()),
            Iter1(data.data() + data.size()),
            Iter2(result.data()),
            init,
            binary_op,
            unary_op);
        assert(ret == Iter2(result.data() + size));
        assert(result == expected);

        // In place
        ret = std::transform_exclusive_scan(
            policy,
            Iter1(data.data()),
            Iter1(data.data() + data.size()),
            Iter2(data.data()),
            init,
            binary_op,
            unary_op);
        assert(ret == Iter2(data.data() + size));
        assert(data == expected);
      }
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
  types::for_each(
      types::forward_iterator_list<const unsigned char*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<std::size_t*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestUnsignedCharToSize_t, Iter>::template apply>{});
      }});
  types::for_each(types::forward_iterator_list<std::string*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<std::string*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestString, Iter>::template apply>{});
                  }});
  return 0;
}
