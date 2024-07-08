//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{10.13|10.14|10.15|11.0}}

// <string>

// This test demonstrates the smaller allocation sizes when the alignment
// requirements of std::string are dropped from 16 to 8.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

#include "test_macros.h"

// alignment of the string heap buffer is hardcoded to 8
const std::size_t alignment = 8;

int main(int, char**) {
  std::string input_string;
  input_string.resize(64, 'a');

  // Call a constructor which selects its size using __recommend.
  std::string test_string(input_string.data());
  const std::size_t expected_align8_size = 71;

  // Demonstrate the lesser capacity/allocation size when the alignment requirement is 8.
  if (alignment == 8) {
    assert(test_string.capacity() == expected_align8_size);
  } else {
    assert(test_string.capacity() == expected_align8_size + 8);
  }

  return 0;
}
