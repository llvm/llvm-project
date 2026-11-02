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
//           class T>
//   ForwardIterator2 inclusive_scan(ExecutionPolicy&& exec,
//                                   ForwardIterator1 first,
//                                   ForwardIterator1 last,
//                                   ForwardIterator2 result,
//                                   BinaryOperation reduce,
//                                   T init);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(inclusive_scan);

static_assert(sfinae_test_inclusive_scan<int, int*, int*, int*, int (*)(int, int), int>);
static_assert(!sfinae_test_inclusive_scan<std::execution::parallel_policy, int*, int*, int*, int (*)(int, int), int>);

template <typename T>
constexpr auto triangle(T n) {
  return n * (n + 1) / 2;
}

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {0};
      auto res =
          std::inclusive_scan(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, 0);
      static_assert(std::is_same_v<decltype(res), Iter2>);
    }
    { // Empty
      int a[] = {-1};
      std::inclusive_scan(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(a)), std::plus<>{}, 10);
      assert(a[0] == -1);
    }
    { // Scan a uniform array
      int a[10];
      std::fill(std::begin(a), std::end(a), 3);
      std::inclusive_scan(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, 50);
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == 50 + (static_cast<int>(i) + 1) * 3);
    }
    { // Scan an increasing array starting from 0
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      std::inclusive_scan(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, 40);
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(40 + triangle(i)));
    }
    { // Scan an increasing array starting from 1
      int a[10];
      std::iota(std::begin(a), std::end(a), 1);
      std::inclusive_scan(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), std::plus<>{}, 30);
      for (std::size_t i = 0; i < std::size(a); ++i)
        assert(a[i] == static_cast<int>(30 + triangle(i + 1)));
    }
    { // Test against pre-computed expected results
      auto test = [&](auto first, auto last, auto reduce, auto init, auto exp_first, auto exp_last) {
        int out[5];
        // Not in place
        auto end = std::inclusive_scan(policy, first, last, Iter2(std::begin(out)), reduce, init);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
        // In place
        std::copy(first, last, out);
        end = std::inclusive_scan(policy, Iter2(std::begin(out)), end, Iter2(std::begin(out)), reduce, init);
        assert(std::equal(Iter2(std::begin(out)), end, exp_first, exp_last));
      };
      const int a[]          = {1, 3, 5, 7, 9};
      const int exp_plus_0[] = {1, 4, 9, 16, 25};           // plus, init = 0
      const int exp_mul_1[]  = {1, 3, 15, 105, 945};        // multiplies, init = 1
      static_assert(std::size(a) == std::size(exp_plus_0)); // just to be sure
      static_assert(std::size(a) == std::size(exp_mul_1));  // just to be sure
      std::plus<> plus;
      std::multiplies<> mult;
      for (unsigned int i = 0; i < std::size(a); ++i) {
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), plus, 0, exp_plus_0, exp_plus_0 + i);
        test(Iter1(std::begin(a)), Iter1(std::begin(a) + i), mult, 1, exp_mul_1, exp_mul_1 + i);
      }
    }
    { // Construct iota from std::plus<>
      int a[1073];
      int b[1073];
      std::fill(std::begin(a), std::end(a), 1);
      runway_sample(std::size(a) + 1, [&](size_t i) {
        Iter2 r = std::inclusive_scan(
            policy, Iter1(std::begin(a)), Iter1(std::begin(a) + i), Iter2(std::begin(b)), std::plus<>{}, 1);
        assert(r == Iter2(std::begin(b) + i));
        for (size_t j = 0; j < i; ++j) {
          assert(b[j] == static_cast<int>(j) + 2);
        }
        std::fill(std::begin(b), std::begin(b) + i, -1);
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
    std::fill(std::begin(a), std::end(a), static_cast<unsigned char>(255));
    runway_sample(std::size(a) + 1, [&](size_t i) {
      std::inclusive_scan(
          policy, Iter1(std::begin(a)), Iter1(std::begin(a) + i), Iter2(std::begin(b)), std::plus<>{}, std::size_t{0});
      for (std::size_t k = 0; k < i; ++k)
        assert(b[k] == 255 * (k + 1));
    });
  }
};

template <class Iter1, class Iter2>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      auto op          = [](const std::string& a, const std::string& b) { return a + b; };
      std::string init = "0";
      for (std::size_t size : {0, 1, 2, 3, 10, 100, 1000}) {
        std::vector<std::string> data(size);
        for (std::size_t i = 0; i < size; ++i)
          data[i] = std::string(1, static_cast<char>('a' + i % 26));

        std::vector<std::string> expected(size);
        std::inclusive_scan(data.begin(), data.end(), expected.begin(), op, init);

        // Not in place
        std::vector<std::string> result(size);
        auto ret = std::inclusive_scan(
            policy, Iter1(data.data()), Iter1(data.data() + data.size()), Iter2(result.data()), op, init);
        assert(ret == Iter2(result.data() + size));
        assert(result == expected);

        // In place
        ret = std::inclusive_scan(
            policy, Iter1(data.data()), Iter1(data.data() + data.size()), Iter2(data.data()), op, init);
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
