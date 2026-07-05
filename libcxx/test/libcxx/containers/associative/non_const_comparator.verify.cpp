//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: diagnose-if-support

// Test that libc++ generates a warning diagnostic when the container is
// provided a non-const callable comparator.

#include <__type_traits/invoke.h>
#include <map>
#include <set>

struct BadCompare {
  template <class T, class U>
  bool operator()(T const& t, U const& u) {
    return t < u;
  }
};

void f() {
  static_assert(!std::__is_invocable_v<BadCompare const&, int const&, int const&>, "");
  static_assert(std::__is_invocable_v<BadCompare&, int const&, int const&>, "");

  // expected-warning@set:* 2 {{the specified comparator type does not provide a viable const call operator}}
  // expected-warning@map:* 2 {{the specified comparator type does not provide a viable const call operator}}
  {
    using C = std::set<int, BadCompare>;
    C s;
  }
  {
    using C = std::multiset<long, BadCompare>;
    C s;
  }
  {
    using C = std::map<int, int, BadCompare>;
    C s;
  }
  {
    using C = std::multimap<long, int, BadCompare>;
    C s;
  }
}
