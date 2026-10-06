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
//           class UnaryPredicate>
//   ForwardIterator2 copy_if(ExecutionPolicy&& exec,
//                            ForwardIterator1 first,
//                            ForwardIterator1 last,
//                            ForwardIterator2 result,
//                            UnaryPredicate pred);

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

EXECUTION_POLICY_SFINAE_TEST(copy_if);

static_assert(sfinae_test_copy_if<int, int*, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_copy_if<std::execution::parallel_policy, int*, int*, int*, bool (*)(int)>);

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    auto always_true  = [](int) { return true; };
    auto always_false = [](int) { return false; };
    { // Check the return types
      int in[]  = {0};
      int out[] = {-1};
      auto res =
          std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::begin(in)), Iter2(std::begin(out)), always_true);
      static_assert(std::is_same_v<decltype(res), Iter2>);
    }
    { // Empty range
      int in[]  = {0};
      int out[] = {-1};
      auto res =
          std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::begin(in)), Iter2(std::begin(out)), always_true);
      assert(res == Iter2(std::begin(out)));
    }
    { // Single, true
      int in[]  = {0};
      int out[] = {-1};
      auto res  = std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::end(in)), Iter2(std::begin(out)), always_true);
      assert(res == Iter2(std::begin(out) + 1));
      assert(out[0] == 0);
    }
    { // Single, false
      int in[]  = {0};
      int out[] = {-1};
      auto res = std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::end(in)), Iter2(std::begin(out)), always_false);
      assert(res == Iter2(std::begin(out)));
      assert(out[0] == -1);
    }
    {
      // Two, true, false
      int in[]  = {0, 1};
      int out[] = {-1, -1};
      auto res  = std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::end(in)), Iter2(std::begin(out)), [](int x) {
        return x == 0;
      });
      assert(res == Iter2(std::begin(out) + 1));
      assert(out[0] == 0);
      assert(out[1] == -1);
    }
    {
      // Two, false, true
      int in[]  = {0, 1};
      int out[] = {-1, -1};
      auto res  = std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::end(in)), Iter2(std::begin(out)), [](int x) {
        return x != 0;
      });
      assert(res == Iter2(std::begin(out) + 1));
      assert(out[0] == 1);
      assert(out[1] == -1);
    }
    { // iotaed input sampled at different lengths, pred: always true
      int in[1073];
      int out[1073];
      std::iota(std::begin(in), std::end(in), 0);
      runway_sample(std::size(in) + 1, [&](size_t n) {
        std::fill(std::begin(out), std::end(out), -1);
        auto res =
            std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::begin(in) + n), Iter2(std::begin(out)), always_true);
        assert(res == Iter2(std::begin(out) + n));
        for (size_t i = 0; i < std::size(out); ++i) {
          assert(out[i] == (i < n ? in[i] : -1));
        }
      });
    }
    { // iotaed input sampled at different lengths, pred: always false
      int in[1073];
      int out[1073];
      std::iota(std::begin(in), std::end(in), 0);
      runway_sample(std::size(in) + 1, [&](size_t n) {
        std::fill(std::begin(out), std::end(out), -1);
        auto res = std::copy_if(
            policy, Iter1(std::begin(in)), Iter1(std::begin(in) + n), Iter2(std::begin(out)), always_false);
        assert(res == Iter2(std::begin(out)));
        for (size_t i = 0; i < std::size(out); ++i) {
          assert(out[i] == -1);
        }
      });
    }
    { // iotaed input sampled at different lengths, pred: x % 3 == 0
      int in[1073];
      int out[1073];
      std::iota(std::begin(in), std::end(in), 0);
      runway_sample(std::size(in) + 1, [&](size_t n) {
        std::fill(std::begin(out), std::end(out), -1);
        auto res =
            std::copy_if(policy, Iter1(std::begin(in)), Iter1(std::begin(in) + n), Iter2(std::begin(out)), [](int x) {
              return x % 3 == 0;
            });
        assert(res == Iter2(std::begin(out) + (n + 2) / 3));
        auto out_count = std::distance(Iter2(std::begin(out)), res);
        for (size_t i = 0; i < std::size(out); ++i) {
          if (i < static_cast<size_t>(out_count)) {
            assert(out[i] % 3 == 0);
            assert(out[i] == (i > 0 ? out[i - 1] + 3 : 0));
          } else {
            assert(out[i] == -1);
          }
        }
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
