//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// Test CTAD on cases where deduction should fail.

#include <flat_map>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

using P  = std::pair<int, long>;
using PC = std::pair<const int, long>;

void test() {
  {
    // cannot deduce Key and T from just (KeyContainer), even if it's a container of pairs
    std::vector<std::pair<int, int>> v;
    std::flat_map s(v);
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce Key and T from just (KeyContainer, Allocator)
    std::vector<int> v;
    std::flat_map s(v, std::allocator<std::pair<const int, int>>());
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce Key and T from nothing
    std::flat_map m;
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce Key and T from just (Compare)
    std::flat_map m(std::less<int>{});
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce Key and T from just (Compare, Allocator)
    std::flat_map m(std::less<int>{}, std::allocator<PC>{});
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce Key and T from just (Allocator)
    std::flat_map m(std::allocator<PC>{});
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot convert from some arbitrary unrelated type
    NotAnAllocator a;
    std::flat_map m(a);
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce that the inner braced things should be std::pair and not something else
    std::flat_map m{{1, 1L}, {2, 2L}, {3, 3L}};
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce that the inner braced things should be std::pair and not something else
    std::flat_map m({{1, 1L}, {2, 2L}, {3, 3L}}, std::less<int>());
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce that the inner braced things should be std::pair and not something else
    std::flat_map m({{1, 1L}, {2, 2L}, {3, 3L}}, std::less<int>(), std::allocator<PC>());
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // cannot deduce that the inner braced things should be std::pair and not something else
    std::flat_map m({{1, 1L}, {2, 2L}, {3, 3L}}, std::allocator<PC>());
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // since we have parens, not braces, this deliberately does not find the initializer_list constructor
    std::flat_map m(P{1, 1L});
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
  {
    // since we have parens, not braces, this deliberately does not find the initializer_list constructor
    std::flat_map m(PC{1, 1L});
    // expected-error-re@-1{{{{no viable constructor or deduction guide for deduction of template arguments of '.*flat_map'}}}}
  }
}
