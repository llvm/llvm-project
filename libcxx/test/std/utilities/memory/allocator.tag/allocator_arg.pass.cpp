//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// struct allocator_arg_t { };
// const allocator_arg_t allocator_arg = allocator_arg_t();

#include <memory>

void test(std::allocator_arg_t) {}

int main(int, char**)
{
    test(std::allocator_arg);

  return 0;
}
