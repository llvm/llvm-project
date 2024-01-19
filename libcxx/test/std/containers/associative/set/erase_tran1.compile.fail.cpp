//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <set>

// class set

// template<class K>
//     size_type erase(K&& x);
//
//   The member function templates find, count, lower_bound, upper_bound,
// equal_range, erase, and extract shall not participate in overload resolution
// unless the qualified-id Compare::is_transparent is valid and denotes a type.
// Additionally, the member function templates erase and extract shall not
// participate in overload resolution if is_convertible_v<K&&, iterator> ||
// is_convertible_v<K&&, const_iterator> is true, where K is the type
// substituted as the first template argument

#include <set>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

int main(int, char**) {
  {
    typedef std::set<int, transparent_less_no_type> M;

    TEST_IGNORE_NODISCARD M().erase(C2Int{5});
  }
}
