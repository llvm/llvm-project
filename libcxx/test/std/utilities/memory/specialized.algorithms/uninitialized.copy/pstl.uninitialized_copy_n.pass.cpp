//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <memory>

// template <class ExecutionPolicy,
//           class InputIterator,
//           class Size,
//           class ForwardIterator>
//   ForwardIterator uninitialized_copy_n(ExecutionPolicy&& exec,
//                                        InputIterator first, Size n,
//                                        ForwardIterator result);

#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(uninitialized_copy_n);

static_assert(sfinae_test_uninitialized_copy_n<int, int*, int, int*>);
static_assert(!sfinae_test_uninitialized_copy_n<std::execution::parallel_policy, int*, int, int*>);

// Source type
struct Src {
  Src()                      = default;
  Src(const Src&)            = delete;
  Src& operator=(const Src&) = delete;
  int value() const { return value_; }
  void set_value(int v) { value_ = v; }

private:
  int value_ = 0;
};

// Destination type
struct Dst {
  Dst(const Src& x) : value_(x.value()) {} // Can only be constructed from Src
  Dst(const Dst&) = delete;
  ~Dst() { value_ = 0; };
  Dst& operator=(const Dst&) = delete;
  int value() const { return value_; }

private:
  int value_;
};

template <class Iter1, class Iter2>
struct TestCustomTypes {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    constexpr size_t n = 1073;

    // Source is Src(1), Src(2), Src(3), ...
    std::array<Src, n> source;
    for (size_t i = 0; i < n; ++i) {
      source[i].set_value(static_cast<int>(i + 1));
    }

    std::allocator<Dst> alloc_dst;
    Dst* dest = alloc_dst.allocate(n);

    // Copy-construct different ranges of Y [0..size) from the source X array
    runway_sample(n + 1, [&](size_t size) {
      auto ret = std::uninitialized_copy_n(policy, Iter1(source.data()), size, Iter2(dest));
      ASSERT_SAME_TYPE(decltype(ret), Iter2);
      assert(ret == Iter2(dest + size));

      for (size_t i = 0; i < size; ++i) {
        assert(dest[i].value() == source[i].value()); // Dest is Dst(1), Dst(2), Dst(3), ...
      }

      std::destroy_n(dest, size); // Clear dest for the next iteration
    });

    alloc_dst.deallocate(dest, n);
  }
};

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      constexpr size_t n = 1073;

      std::array<int, n> source;
      std::iota(source.begin(), source.end(), 1); // Source is 1 2 3 4 ...

      std::allocator<int> alloc;
      int* dest = alloc.allocate(n);
      std::fill(dest, dest + n, 0); // Dest is 0 0 0 0 ...

      runway_sample(n + 1, [&](size_t size) {
        auto ret = std::uninitialized_copy_n(policy, Iter1(source.data()), size, Iter2(dest));
        ASSERT_SAME_TYPE(decltype(ret), Iter2);
        assert(ret == Iter2(dest + size));

        for (size_t i = 0; i < size; ++i) {
          assert(dest[i] == source[i]); // Dest is 1 2 3 4 ...
        }

        std::fill(dest, dest + size, 0); // Clear dest for the next iteration
      });

      alloc.deallocate(dest, n);
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::cpp17_input_iterator_list<const Src*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<Dst*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestCustomTypes, Iter>::template apply>{});
      }});
  types::for_each(types::cpp17_input_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});
  return 0;
}
