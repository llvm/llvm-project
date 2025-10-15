//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// Make sure that we don't error out when passing a comparator that is lvalue callable
// but not rvalue callable to algorithms. While it is technically ill-formed for users
// to provide us such predicates, this test is useful for libc++ to ensure that we check
// predicate requirements correctly (i.e. that we check them on lvalues and not on
// rvalues). See https://llvm.org/PR69554 for additional context.

#include <algorithm>

#include "test_macros.h"

struct NotRvalueCallable {
  bool operator()(int a, int b) const& { return a < b; }
  bool operator()(int, int) && = delete;
};

void f() {
  int a[] = {1, 2, 3, 4};
  (void)std::lower_bound(a, a + 4, 0, NotRvalueCallable{});
  (void)std::upper_bound(a, a + 4, 0, NotRvalueCallable{});
  (void)std::minmax({1, 2, 3}, NotRvalueCallable{});
  (void)std::minmax_element(a, a + 4, NotRvalueCallable{});
  (void)std::min_element(a, a + 4, NotRvalueCallable{});
  (void)std::max_element(a, a + 4, NotRvalueCallable{});
  (void)std::is_permutation(a, a + 4, a, NotRvalueCallable{});
#if TEST_STD_VER >= 14
  (void)std::is_permutation(a, a + 4, a, a + 4, NotRvalueCallable{});
#endif
  (void)std::includes(a, a + 4, a, a + 4, NotRvalueCallable{});
  (void)std::equal_range(a, a + 4, 0, NotRvalueCallable{});
  (void)std::partial_sort_copy(a, a + 4, a, a + 4, NotRvalueCallable{});
  (void)std::search(a, a + 4, a, a + 4, NotRvalueCallable{});
  (void)std::search_n(a, a + 4, 4, 0, NotRvalueCallable{});
}
