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

// <semaphore>

#include <semaphore>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  std::counting_semaphore<> s(0);

  s.release();
  s.acquire();

  std::thread t = support::make_test_thread([&](){
    s.acquire();
  });
  s.release(2);
  t.join();
  s.acquire();

  return 0;
}
