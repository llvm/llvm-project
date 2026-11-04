//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// The binary-search algorithms validate the whole input range once, up front, when they are given libc++'s bounded
// iterators, and then search using the underlying pointers. Make sure invalid ranges are still diagnosed.

// REQUIRES: can-test-hardening-assertions-fast
// Continuing past a failed check is undefined behavior here.
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

#include <__iterator/bounded_iter.h>
#include <__iterator/static_bounded_iter.h>
#include <algorithm>
#include <functional>
#include <vector>

#include "check_assertion.h"
#include "test_macros.h"

int main(int, char**) {
  int array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // __bounded_iter: a reversed range.
  {
    auto first = std::__make_bounded_iter(array + 5, array, array + 10);
    auto last  = std::__make_bounded_iter(array + 2, array, array + 10);
    TEST_LIBCPP_ASSERT_FAILURE(std::lower_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::upper_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::equal_range(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::binary_search(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::lower_bound(first, last, 3, std::less<int>()),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
#if TEST_STD_VER >= 20
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::lower_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::upper_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::equal_range(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::binary_search(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
#endif
  }

  // __bounded_iter: `last` lies past the end of `first`'s bounds (e.g. it comes from a different range).
  {
    auto first = std::__make_bounded_iter(array, array, array + 5);
    auto last  = std::__make_bounded_iter(array + 8, array, array + 10);
    TEST_LIBCPP_ASSERT_FAILURE(std::lower_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range that extends past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::upper_bound(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range that extends past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::equal_range(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range that extends past the end");
    TEST_LIBCPP_ASSERT_FAILURE(std::binary_search(first, last, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range that extends past the end");
  }

  // __static_bounded_iter: a reversed range, and a range extending past the end.
  {
    auto first = std::__make_static_bounded_iter<10>(array + 5, array);
    auto last  = std::__make_static_bounded_iter<10>(array + 2, array);
    TEST_LIBCPP_ASSERT_FAILURE(
        std::lower_bound(first, last, 3),
        "__static_bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::equal_range(first, last, 3),
        "__static_bounded_iter::__validate_range: Attempt to use a range with a negative length");

    // `last` is valid in its own window, [array + 5, array + 10], but lies past the end of `short_first`'s.
    auto short_first = std::__make_static_bounded_iter<5>(array, array);
    auto other_last  = std::__make_static_bounded_iter<5>(array + 8, array + 5);
    TEST_LIBCPP_ASSERT_FAILURE(
        std::upper_bound(short_first, other_last, 3),
        "__static_bounded_iter::__validate_range: Attempt to use a range that extends past the end");
  }

#ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_VECTOR
  // The same through std::vector's iterators.
  {
    std::vector<int> v(array, array + 10);
    TEST_LIBCPP_ASSERT_FAILURE(std::lower_bound(v.end(), v.begin(), 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
    TEST_LIBCPP_ASSERT_FAILURE(std::binary_search(v.begin() + 6, v.begin() + 1, 3),
                               "__bounded_iter::__validate_range: Attempt to use a range with a negative length");
  }
#endif

  return 0;
}
