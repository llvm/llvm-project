//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast

// <valarray>

// template<class T> class indirect_array;

// const indirect_array& operator=(const indirect_array& ia) const; // where ia has a different size

#include <cstddef>
#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  std::valarray<int> array(1, 16);
  std::size_t target_indices[]    = {0, 2, 4};
  std::size_t same_size_indices[] = {1, 3, 5};
  std::size_t longer_indices[]    = {1, 3, 5, 7};
  std::indirect_array<int> target = array[std::valarray<std::size_t>(target_indices, 3)];

  // Check that there's no assertion for arrays of the same size.
  target = array[std::valarray<std::size_t>(same_size_indices, 3)];

  TEST_LIBCPP_ASSERT_FAILURE(
      target = array[std::valarray<std::size_t>(longer_indices, 4)], "indirect_array::operator= size mismatch");

  return 0;
}
