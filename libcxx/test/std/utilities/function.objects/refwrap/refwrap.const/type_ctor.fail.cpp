//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// reference_wrapper(T&&) = delete;

// XFAIL: c++98, c++03

#include <functional>
#include <cassert>

int main(int, char**)
{
    std::reference_wrapper<const int> r(3);

  return 0;
}
