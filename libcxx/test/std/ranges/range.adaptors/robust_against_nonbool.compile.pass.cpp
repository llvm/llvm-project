//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ranges>
//
// Range adaptors that take predicates should support predicates that return a non-boolean
// value as long as the returned type is implicitly convertible to bool.

#include <ranges>

#include "boolean_testable.h"

template <std::ranges::view View>
constexpr void use(View view) {
  // Just use the view in a few ways. Our goal here is to trigger the instantiation
  // of various functions related to the view and its iterators in the hopes that we
  // instantiate functions that might have incorrect implementations w.r.t. predicates.
  auto first = std::ranges::begin(view);
  auto last  = std::ranges::end(view);
  ++first;
  --first;
  (void)(first == last);
  (void)(first != last);
  (void)std::ranges::empty(view);
}

using Value    = StrictComparable<int>;
using Iterator = StrictBooleanIterator<Value*>;
using Range    = std::ranges::subrange<Iterator>;
auto pred1     = StrictUnaryPredicate;
auto pred2     = StrictBinaryPredicate;

void f(Range in) {
  (void)pred1;
  (void)pred2;

#if TEST_STD_VER >= 23
  {
    auto view = std::views::chunk_by(in, pred2);
    use(view);
  }
#endif
  {
    auto view = std::views::drop_while(in, pred1);
    use(view);
  }
  {
    auto view = std::views::filter(in, pred1);
    use(view);
  }
  {
    auto view = std::views::take_while(in, pred1);
    use(view);
  }
}
