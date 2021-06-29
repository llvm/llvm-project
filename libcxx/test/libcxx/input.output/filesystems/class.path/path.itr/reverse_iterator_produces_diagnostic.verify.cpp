//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

#include "filesystem_include.h"
#include <iterator>


int main(int, char**) {
  using namespace fs;
  using RIt = std::reverse_iterator<path::iterator>;

  // expected-error-re@*:* {{static_assert failed{{.*}} "The specified iterator type cannot be used with reverse_iterator; Using stashing iterators with reverse_iterator causes undefined behavior"}}
  {
    RIt r;
    ((void)r);
  }

  return 0;
}
