//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++98 || c++03 || c++11 || c++14

// test unexpected

#include <exception>
#include <cstdlib>
#include <cassert>

void f1()
{
    std::exit(0);
}

int main(int, char**)
{
    std::set_unexpected(f1);
    std::unexpected();
    assert(false);

  return 0;
}
