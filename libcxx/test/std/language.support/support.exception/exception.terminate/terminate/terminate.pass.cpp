//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test terminate

#include <exception>
#include <cstdlib>
#include <cassert>

void f1()
{
    std::exit(0);
}

int main()
{
    std::set_terminate(f1);
    std::terminate();
    assert(false);
}
