//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit iterator(Parent& parent)
//   requires (!forward_range<Base>); // exposition only

#include <string>
#include <ranges>

#include "types.h"

constexpr bool test() {
  std::string strings[4] = {"eeee", "ffff", "gggg", "hhhh"};

  MoveOnAccessSubrange r{
      DieOnCopyIterator(cpp20_input_iterator(strings)), sentinel_wrapper(cpp20_input_iterator(strings + 4))};
  std::ranges::join_view jv(std::move(r));
  auto iter = jv.begin(); // Calls `iterator(Parent& parent)`
  assert(*iter == 'e');

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
