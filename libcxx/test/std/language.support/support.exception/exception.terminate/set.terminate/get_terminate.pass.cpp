//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test get_terminate

#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

void f1() {}
void f2() {}

int main(int, char**)
{
    std::set_terminate(f1);
    assert(std::get_terminate() == f1);
    std::set_terminate(f2);
    assert(std::get_terminate() == f2);

  return 0;
}
