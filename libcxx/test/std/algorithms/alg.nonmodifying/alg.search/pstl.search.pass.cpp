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

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   ForwardIterator1
//   search(ExecutionPolicy&& exec,
//            ForwardIterator1 first1, ForwardIterator1 last1,
//            ForwardIterator2 first2, ForwardIterator2 last2);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iterator>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(search);

static_assert(sfinae_test_search<int, int*, int*, int*, int*>);
static_assert(!sfinae_test_search<std::execution::parallel_policy, int*, int*, int*, int*>);

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {0};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      static_assert(std::is_same_v<decltype(res), Iter1>);
    }
    { // Empty haystack, empty needle
      int a[] = {0};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(b)), Iter2(std::begin(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Empty haystack, non-empty needle
      int a[] = {0};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Non-empty haystack, empty needle
      int a[] = {0};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::begin(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Both single element, same
      int a[] = {0};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Both single element, different
      int a[] = {0};
      int b[] = {1};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Needle found at beginning
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Needle found in middle
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {1};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 1));
    }
    { // Needle found at end
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {5};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a) - 1));
    }
    { // Multiple element needle found at beginning
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {0, 1, 2, 3, 4, 5};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Multiple element needle found in middle
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {2};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 2));
    }
    { // Multiple element needle found at end
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {3, 4, 5};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 3));
    }
    { // Needle too long for haystack
      int a[] = {0, 1, 2, 3, 4};
      int b[] = {0, 1, 2, 3, 4, 5};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Needle much longer than haystack
      int a[] = {0};
      int b[] = {0, 1, 2, 3, 4, 5};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Multiple needles with size=1, the first one is found
      int a[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[] = {1};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 1));
    }
    { // Multiple needles with size=2, the first one is found
      int a[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[] = {1, 2};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 1));
    }
    { // Multiple needles with size=3, the first one is found
      int a[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[] = {1, 2, 3};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 4));
    }
    { // Single needle with size=4, found at end
      int a[] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[] = {1, 2, 3, 4};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 8));
    }
    { // Needle with repeated elements in the prefix
      int a[] = {0, 1, 1, 1, 1, 2, 3, 0, 1, 2, 3, 4};
      int b[] = {1, 1, 2};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 3));
    }
    { // Long needle, found at the end
      int a[] = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
      int b[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 6));
    }
    { // Needle not found - no match exists
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {6};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Needle not found, size=3
      int a[] = {0, 1, 2, 3, 4, 5};
      int b[] = {4, 5, 6};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Partial match, fails at last element
      int a[] = {0, 1, 2, 3, 4, 5, 6};
      int b[] = {3, 4, 6};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // Partial match, fails in middle
      int a[] = {0, 1, 2, 4, 5};
      int b[] = {1, 3, 4};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::end(a)));
    }
    { // All elements identical in haystack
      int a[] = {7, 7, 7, 7, 7, 7};
      int b[] = {7, 7};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a)));
    }
    { // Multiple partial matches before real match
      int a[] = {1, 2, 1, 2, 1, 2, 3};
      int b[] = {1, 2, 3};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 4));
    }
    { // Alternating pattern
      int a[] = {0, 1, 0, 1, 0, 1, 0, 1};
      int b[] = {1, 0, 1, 0};
      auto res =
          std::search(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)));
      assert(res == Iter1(std::begin(a) + 1));
    }
    { // Many elements, needles found at various sampled positions
      int a[1073];
      std::fill(std::begin(a), std::end(a), 0);
      int needle1[] = {1};
      int needle2[] = {1, 0, 0};
      int needle3[] = {1, 0, 0, 0, 0};
      runway_sample(std::size(a) - std::size(needle3) + 1, [&](std::size_t i) {
        a[i] = 1;
        assert(std::search(policy,
                           Iter1(std::begin(a)),
                           Iter1(std::end(a)),
                           Iter2(std::begin(needle1)),
                           Iter2(std::end(needle1))) == Iter1(std::begin(a) + i));
        assert(std::search(policy,
                           Iter1(std::begin(a)),
                           Iter1(std::end(a)),
                           Iter2(std::begin(needle2)),
                           Iter2(std::end(needle2))) == Iter1(std::begin(a) + i));
        assert(std::search(policy,
                           Iter1(std::begin(a)),
                           Iter1(std::end(a)),
                           Iter2(std::begin(needle3)),
                           Iter2(std::end(needle3))) == Iter1(std::begin(a) + i));
        a[i] = 0;
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies< types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
