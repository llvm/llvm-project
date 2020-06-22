//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14

// <filesystem>

// namespace std::filesystem

#include <filesystem>
#include "test_macros.h"

using namespace std::filesystem;

#if TEST_STD_VER >= 11
// expected-error@-3 {{no namespace named 'filesystem' in namespace 'std';}}
#else
// expected-error@-5 {{expected namespace name}}
#endif

int main(int, char**) {


  return 0;
}
