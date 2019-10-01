//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Class bad_function_call

// class bad_function_call
//     : public exception
// {
// public:
//   // 20.7.16.1.1, constructor:
//   bad_function_call();
// };

#include <functional>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::exception, std::bad_function_call>::value), "");

  return 0;
}
