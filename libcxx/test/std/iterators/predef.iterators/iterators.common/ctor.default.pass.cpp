//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr common_iterator() requires default_initializable<I> = default;

#include <iterator>
#include <cassert>

#include "test_iterators.h"

constexpr bool test()
{
  {
    using It = cpp17_input_iterator<int*>;
    using CommonIt = std::common_iterator<It, sentinel_wrapper<It>>;
    static_assert(!std::is_default_constructible_v<It>); // premise
    static_assert(!std::is_default_constructible_v<CommonIt>); // conclusion
  }
  {
    // The base iterator is value-initialized.
    std::common_iterator<int*, sentinel_wrapper<int*>> c;
    assert(c == nullptr);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
