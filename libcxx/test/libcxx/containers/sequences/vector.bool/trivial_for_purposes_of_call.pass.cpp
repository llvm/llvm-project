//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// typedef ... iterator;
// typedef ... const_iterator;

// The libc++ __bit_iterator type has weird ABI calling conventions as a quirk
// of the implementation. The const bit iterator is trivial, but the non-const
// bit iterator is not because it declares a user-defined copy constructor.
//
// Changing this now is an ABI break, so this test ensures that each type
// is trivial/non-trivial as expected.

// The definition of 'non-trivial for the purposes of calls':
//   A type is considered non-trivial for the purposes of calls if:
//     * it has a non-trivial copy constructor, move constructor, or
//       destructor, or
//     * all of its copy and move constructors are deleted.

// UNSUPPORTED: c++98, c++03

#include <vector>
#include <cassert>

#include "test_macros.h"

template <class T>
using IsTrivialForCall = std::integral_constant<bool,
  std::is_trivially_copy_constructible<T>::value &&
  std::is_trivially_move_constructible<T>::value &&
  std::is_trivially_destructible<T>::value
  // Ignore the all-deleted case, it shouldn't occur here.
  >;

void test_const_iterator() {
  using It = std::vector<bool>::const_iterator;
  static_assert(IsTrivialForCall<It>::value, "");
}

void test_non_const_iterator() {
  using It = std::vector<bool>::iterator;
  static_assert(!IsTrivialForCall<It>::value, "");
}

int main(int, char**) {
  test_const_iterator();
  test_non_const_iterator();

  return 0;
}
