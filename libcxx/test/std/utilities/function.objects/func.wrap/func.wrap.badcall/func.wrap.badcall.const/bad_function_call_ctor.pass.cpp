//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Class bad_function_call

// bad_function_call();

#include <functional>
#include <type_traits>

int main(int, char**)
{
    std::bad_function_call ex;

  return 0;
}
