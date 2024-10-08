//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Ignore spurious errors after the initial static_assert failure.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// <algorithm>

// Check that calling a classic STL algorithm with various non-callable comparators is diagnosed.

#include <algorithm>

#include "test_macros.h"

struct NotCallable {
  bool compare(int a, int b) const { return a < b; }
};

struct NotMutableCallable {
  bool operator()(int a, int b) = delete;
  bool operator()(int a, int b) const { return a < b; }
};

void f() {
  int a[] = {1, 2, 3, 4};
  {
    (void)std::lower_bound(a, a + 4, 0, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::upper_bound(a, a + 4, 0, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::minmax({1, 2, 3}, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::minmax_element(a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::min_element(a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::max_element(a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::is_permutation(a, a + 4, a, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

#if TEST_STD_VER >= 14
    (void)std::is_permutation(a, a + 4, a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}
#endif

    (void)std::includes(a, a + 4, a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::equal_range(a, a + 4, 0, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::partial_sort_copy(a, a + 4, a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::search(a, a + 4, a, a + 4, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::search_n(a, a + 4, 4, 0, &NotCallable::compare);
    // expected-error@*:* {{The comparator has to be callable}}
  }

  {
    (void)std::lower_bound(a, a + 4, 0, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::upper_bound(a, a + 4, 0, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::minmax({1, 2, 3}, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::minmax_element(a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::min_element(a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::max_element(a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::is_permutation(a, a + 4, a, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

#if TEST_STD_VER >= 14
    (void)std::is_permutation(a, a + 4, a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}
#endif

    (void)std::includes(a, a + 4, a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::equal_range(a, a + 4, 0, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::partial_sort_copy(a, a + 4, a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::search(a, a + 4, a, a + 4, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}

    (void)std::search_n(a, a + 4, 4, 0, NotMutableCallable{});
    // expected-error@*:* {{The comparator has to be callable}}
  }
}
