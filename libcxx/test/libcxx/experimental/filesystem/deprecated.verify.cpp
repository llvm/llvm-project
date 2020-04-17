//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support
// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

#include <experimental/filesystem>

using namespace std::experimental::filesystem; // expected-warning {{'filesystem' is deprecated: std::experimental::filesystem has now been deprecated in favor of C++17's std::filesystem. Please stop using it and start using std::filesystem. This experimental version will be removed in LLVM 11. You can remove this warning by defining the _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM macro.}}

int main(int, char**) {
  return 0;
}
