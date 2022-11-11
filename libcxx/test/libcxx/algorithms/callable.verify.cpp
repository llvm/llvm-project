//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// check that the classical algorithms with non-callable comparators fail

#include <algorithm>

void f() {
  struct S {
    int i;

    S(int i_) : i(i_) {}

    bool compare(const S&) const;
  };

  S a[] = {1, 2, 3, 4};
  (void) std::lower_bound(a, a + 4, 0, &S::compare); // expected-error@*:* {{The comparator has to be callable}}
  (void) std::minmax({S{1}}, &S::compare); // expected-error@*:* {{The comparator has to be callable}}
  (void) std::minmax_element(a, a + 4, &S::compare); // expected-error@*:* {{The comparator has to be callable}}
}
