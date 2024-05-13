//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::end

#include <ranges>

struct NonBorrowedRange {
  int* begin() const;
  int* end() const;
};
static_assert(!std::ranges::enable_borrowed_range<NonBorrowedRange>);

// Verify that if the expression is an rvalue and `enable_borrowed_range` is false, `ranges::end` is ill-formed.
void test() {
  std::ranges::end(NonBorrowedRange());
  // expected-error-re@-1 {{{{call to deleted function call operator in type 'const (std::ranges::)?__end::__fn'}}}}
  // expected-error@-2  {{attempt to use a deleted function}}
}
