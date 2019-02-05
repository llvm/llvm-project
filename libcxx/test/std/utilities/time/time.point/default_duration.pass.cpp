//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// Test default template arg:

// template <class Clock, class Duration = typename Clock::duration>
//   class time_point;

#include <chrono>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_same<std::chrono::system_clock::duration,
                   std::chrono::time_point<std::chrono::system_clock>::duration>::value), "");

  return 0;
}
