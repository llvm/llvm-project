//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// Test default template arg:

// template <class Rep, class Period = ratio<1>>
// class duration;

#include <chrono>
#include <ratio>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::chrono::duration<int, std::ratio<1> >,
                   std::chrono::duration<int> >::value), "");

  return 0;
}
