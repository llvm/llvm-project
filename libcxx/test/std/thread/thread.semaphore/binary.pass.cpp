//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// XFAIL: availability-synchronization_library-missing

// <semaphore>

#include <semaphore>
#include <chrono>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(std::is_same<std::binary_semaphore, std::counting_semaphore<1>>::value, "");

int main(int, char**)
{
  std::binary_semaphore s(1);

  auto l = [&](){
    for(int i = 0; i < 1024; ++i) {
        s.acquire();
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        s.release();
    }
  };

  std::thread t = support::make_test_thread(l);
  l();

  t.join();

  return 0;
}
