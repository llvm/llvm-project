//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test set_new_handler

#include <new>
#include <cassert>

void f1() {}
void f2() {}

int main()
{
    assert(std::set_new_handler(f1) == 0);
    assert(std::set_new_handler(f2) == f1);
}
