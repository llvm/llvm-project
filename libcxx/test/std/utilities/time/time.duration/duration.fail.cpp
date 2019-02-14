//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// If a program instantiates duration with a duration type for the template
// argument Rep a diagnostic is required.

#include <chrono>

int main(int, char**)
{
    typedef std::chrono::duration<std::chrono::milliseconds> D;
    D d;

  return 0;
}
