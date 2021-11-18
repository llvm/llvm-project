//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// std::atomic

// atomic& operator=( const atomic& ) volatile = delete;

#include <atomic>

int main(int, char**)
{
  volatile std::atomic<int> obj1;
  std::atomic<int> obj2;
  obj1 = obj2; // expected-error {{overload resolution selected deleted operator '='}}
}
