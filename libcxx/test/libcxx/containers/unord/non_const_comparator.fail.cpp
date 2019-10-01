//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// REQUIRES: diagnose-if-support, verify-support

// Test that libc++ generates a warning diagnostic when the container is
// provided a non-const callable comparator or a non-const hasher.

#include <unordered_set>
#include <unordered_map>

struct BadHash {
  template <class T>
  size_t operator()(T const& t) {
    return std::hash<T>{}(t);
  }
};

struct BadEqual {
  template <class T, class U>
  bool operator()(T const& t, U const& u) {
    return t == u;
  }
};

int main(int, char**) {
  static_assert(!std::__invokable<BadEqual const&, int const&, int const&>::value, "");
  static_assert(std::__invokable<BadEqual&, int const&, int const&>::value, "");

  // expected-warning@unordered_set:* 2 {{the specified comparator type does not provide a viable const call operator}}
  // expected-warning@unordered_map:* 2 {{the specified comparator type does not provide a viable const call operator}}
  // expected-warning@unordered_set:* 2 {{the specified hash functor does not provide a viable const call operator}}
  // expected-warning@unordered_map:* 2 {{the specified hash functor does not provide a viable const call operator}}

  {
    using C = std::unordered_set<int, BadHash, BadEqual>;
    C s;
  }
  {
    using C = std::unordered_multiset<long, BadHash, BadEqual>;
    C s;
  }
  {
    using C = std::unordered_map<int, int, BadHash, BadEqual>;
    C s;
  }
  {
    using C = std::unordered_multimap<long, int, BadHash, BadEqual>;
    C s;
  }

  return 0;
}
