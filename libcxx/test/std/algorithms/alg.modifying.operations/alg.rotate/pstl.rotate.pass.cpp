//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator>
//   ForwardIterator
//     rotate(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator middle, ForwardIterator last);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(rotate);

static_assert(sfinae_test_rotate<int, int*, int*, int*>);
static_assert(!sfinae_test_rotate<std::execution::parallel_policy, int*, int*, int*>);

template <class Iter>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {0};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      static_assert(std::is_same_v<decltype(r), Iter>);
    }
    { // Check empty
      int a[] = {0};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::begin(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
    }
    { // Check single element, rotate at first
      int a[] = {0};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
    }
    { // Check single element, rotate at last
      int a[] = {0};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
    }
    { // Check two elements, rotate at first
      int a[] = {0, 1};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
    }
    { // Check two elements, rotate at second
      int a[] = {0, 1};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == 1);
      assert(a[1] == 0);
    }
    { // Check two elements, rotate at last
      int a[] = {0, 1};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
    }
    { // Check three elements, rotate at first
      int a[] = {0, 1, 2};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
    }
    { // Check three elements, rotate at second
      int a[] = {0, 1, 2};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 2));
      assert(a[0] == 1);
      assert(a[1] == 2);
      assert(a[2] == 0);
    }
    { // Check three elements, rotate at third
      int a[] = {0, 1, 2};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == 2);
      assert(a[1] == 0);
      assert(a[2] == 1);
    }
    { // Check three elements, rotate at last
      int a[] = {0, 1, 2};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
    }
    { // Check four elements, rotate at first
      int a[] = {0, 1, 2, 3};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
    }
    { // Check four elements, rotate at second
      int a[] = {0, 1, 2, 3};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 3));
      assert(a[0] == 1);
      assert(a[1] == 2);
      assert(a[2] == 3);
      assert(a[3] == 0);
    }
    { // Check four elements, rotate at third
      int a[] = {0, 1, 2, 3};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 2));
      assert(a[0] == 2);
      assert(a[1] == 3);
      assert(a[2] == 0);
      assert(a[3] == 1);
    }
    { // Check four elements, rotate at fourth
      int a[] = {0, 1, 2, 3};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 3), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == 3);
      assert(a[1] == 0);
      assert(a[2] == 1);
      assert(a[3] == 2);
    }
    { // Check four elements, rotate at last
      int a[] = {0, 1, 2, 3};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
    }
    { // Check five elements, rotate at first
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
      assert(a[4] == 4);
    }
    { // Check five elements, rotate at second
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 4));
      assert(a[0] == 1);
      assert(a[1] == 2);
      assert(a[2] == 3);
      assert(a[3] == 4);
      assert(a[4] == 0);
    }
    { // Check five elements, rotate at third
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 3));
      assert(a[0] == 2);
      assert(a[1] == 3);
      assert(a[2] == 4);
      assert(a[3] == 0);
      assert(a[4] == 1);
    }
    { // Check five elements, rotate at fourth
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 3), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 2));
      assert(a[0] == 3);
      assert(a[1] == 4);
      assert(a[2] == 0);
      assert(a[3] == 1);
      assert(a[4] == 2);
    }
    { // Check five elements, rotate at fifth
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 4), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == 4);
      assert(a[1] == 0);
      assert(a[2] == 1);
      assert(a[3] == 2);
      assert(a[4] == 3);
    }
    { // Check five elements, rotate at last
      int a[] = {0, 1, 2, 3, 4};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
      assert(a[4] == 4);
    }
    { // Check six elements, rotate at first
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
      assert(a[4] == 4);
      assert(a[5] == 5);
    }
    { // Check six elements, rotate at second
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 5));
      assert(a[0] == 1);
      assert(a[1] == 2);
      assert(a[2] == 3);
      assert(a[3] == 4);
      assert(a[4] == 5);
      assert(a[5] == 0);
    }
    { // Check six elements, rotate at third
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 4));
      assert(a[0] == 2);
      assert(a[1] == 3);
      assert(a[2] == 4);
      assert(a[3] == 5);
      assert(a[4] == 0);
      assert(a[5] == 1);
    }
    { // Check six elements, rotate at fourth
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 3), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 3));
      assert(a[0] == 3);
      assert(a[1] == 4);
      assert(a[2] == 5);
      assert(a[3] == 0);
      assert(a[4] == 1);
      assert(a[5] == 2);
    }
    { // Check six elements, rotate at fifth
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 4), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 2));
      assert(a[0] == 4);
      assert(a[1] == 5);
      assert(a[2] == 0);
      assert(a[3] == 1);
      assert(a[4] == 2);
      assert(a[5] == 3);
    }
    { // Check six elements, rotate at sixth
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 5), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == 5);
      assert(a[1] == 0);
      assert(a[2] == 1);
      assert(a[3] == 2);
      assert(a[4] == 3);
      assert(a[5] == 4);
    }
    { // Check six elements, rotate at sixth
      int a[] = {0, 1, 2, 3, 4, 5};
      auto r  = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
      assert(a[2] == 2);
      assert(a[3] == 3);
      assert(a[4] == 4);
      assert(a[5] == 5);
    }
  }
};

template <class Iter>
struct TestNonCopyable {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      std::unique_ptr<int> a[] = {std::make_unique<int>(0)};
      auto r                   = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      static_assert(std::is_same_v<decltype(r), Iter>);
    }
    { // Check two elements, rotate at first
      std::unique_ptr<int> a[] = {std::make_unique<int>(0), std::make_unique<int>(1)};
      auto r                   = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(r == Iter(std::end(a)));
      assert(*a[0] == 0);
      assert(*a[1] == 1);
    }
    { // Check two elements, rotate at second
      std::unique_ptr<int> a[] = {std::make_unique<int>(0), std::make_unique<int>(1)};
      auto r                   = std::rotate(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(r == Iter(std::begin(a) + 1));
      assert(*a[0] == 1);
      assert(*a[1] == 0);
    }
    { // Check two elements, rotate at last
      std::unique_ptr<int> a[] = {std::make_unique<int>(0), std::make_unique<int>(1)};
      auto r                   = std::rotate(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(r == Iter(std::begin(a)));
      assert(*a[0] == 0);
      assert(*a[1] == 1);
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<TestInt>{});
  types::for_each(types::forward_iterator_list<std::unique_ptr<int>*>{}, TestIteratorWithPolicies<TestNonCopyable>{});
  return 0;
}
