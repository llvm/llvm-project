//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// decltype(auto) operator->() const
//   requires see below;

#include <iterator>
#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

void test() {
  // Case 1: http://eel.is/c++draft/iterators.common#common.iter.access-5.1
  {
    auto check = []<class Iterator>() {
      int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
      Iterator iter(buffer);
      using Common = std::common_iterator<Iterator, sentinel_wrapper<Iterator>>;

      Common common(iter);
      std::same_as<Iterator> auto result = common.operator->();
      assert(base(result) == buffer);

      Common const ccommon(iter);
      std::same_as<Iterator> auto cresult = ccommon.operator->();
      assert(base(cresult) == buffer);
    };

    check.operator()<contiguous_iterator<int*>>();
    check.operator()<int*>();
  }

  // Case 2: http://eel.is/c++draft/iterators.common#common.iter.access-5.2
  {
    auto check = []<class Iterator>() {
      int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
      Iterator iter(buffer);
      using Common = std::common_iterator<Iterator, sentinel_type<int*>>;

      Common common(iter);
      std::same_as<int*> auto result = common.operator->();
      assert(result == buffer);

      Common const ccommon(iter);
      std::same_as<int*> auto cresult = ccommon.operator->();
      assert(cresult == buffer);
    };

    check.operator()<simple_iterator<int*>>();
    check.operator()<cpp17_input_iterator<int*>>();
    // cpp20_input_iterator can't be used with common_iterator because it's not copyable
    check.operator()<forward_iterator<int*>>();
    check.operator()<bidirectional_iterator<int*>>();
    check.operator()<random_access_iterator<int*>>();
  }

  // Case 3: http://eel.is/c++draft/iterators.common#common.iter.access-5.3
  {
    auto check = []<class Iterator>() {
      int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
      Iterator iter(buffer);
      using Common = std::common_iterator<Iterator, sentinel_type<int*>>;

      Common common(iter);
      auto proxy = common.operator->();
      std::same_as<int const*> auto result = proxy.operator->();
      assert(result != buffer); // we copied to a temporary proxy
      assert(*result == *buffer);

      Common const ccommon(iter);
      auto cproxy = ccommon.operator->();
      std::same_as<int const*> auto cresult = cproxy.operator->();
      assert(cresult != buffer); // we copied to a temporary proxy
      assert(*cresult == *buffer);
    };

    check.operator()<value_iterator<int*>>();
    check.operator()<void_plus_plus_iterator<int*>>();
  }
}

int main(int, char**) {
  test();

  return 0;
}
