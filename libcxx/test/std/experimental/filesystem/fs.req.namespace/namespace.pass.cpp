//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <experimental/filesystem>

// namespace std::experimental::filesystem::v1

#include <experimental/filesystem>
#include <type_traits>

#include "test_macros.h"

int main(int, char**) {
  static_assert(std::is_same<
          std::experimental::filesystem::path,
          std::experimental::filesystem::v1::path
      >::value, "");

  return 0;
}
