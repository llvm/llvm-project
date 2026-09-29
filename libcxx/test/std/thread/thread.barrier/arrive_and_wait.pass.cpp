//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <barrier>

#include <barrier>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  std::barrier<> b(2);

  std::thread t = support::make_test_thread([&](){
    for(int i = 0; i < 10; ++i)
      b.arrive_and_wait();
  });
  for(int i = 0; i < 10; ++i)
    b.arrive_and_wait();
  t.join();

  return 0;
}
