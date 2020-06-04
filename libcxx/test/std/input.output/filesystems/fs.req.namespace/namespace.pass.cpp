//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// namespace std::filesystem

#include <filesystem>
#include <type_traits>

#include "test_macros.h"

using namespace std::filesystem;

int main(int, char**) {
  static_assert(std::is_same<
          path,
          std::filesystem::path
      >::value, "");

  return 0;
}
