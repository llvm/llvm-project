//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11

// <latch>

#include <latch>
#include <thread>

#include "test_macros.h"

int main(int, char**)
{
  std::latch l(2);

  l.count_down();
  std::thread t([&](){
    l.count_down();
  });
  l.wait();
  t.join();

  return 0;
}
