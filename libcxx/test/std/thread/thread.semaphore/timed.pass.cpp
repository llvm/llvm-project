//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11

// This test requires the dylib support introduced in D68480, which shipped in
// macOS 11.0.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// <semaphore>

#include <semaphore>
#include <thread>
#include <chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  auto const start = std::chrono::steady_clock::now();

  std::counting_semaphore<> s(0);

  assert(!s.try_acquire_until(start + std::chrono::milliseconds(250)));
  assert(!s.try_acquire_for(std::chrono::milliseconds(250)));

  std::thread t = support::make_test_thread([&](){
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    s.release();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    s.release();
  });

  assert(s.try_acquire_until(start + std::chrono::seconds(2)));
  assert(s.try_acquire_for(std::chrono::seconds(2)));
  t.join();

  auto const end = std::chrono::steady_clock::now();
  assert(end - start < std::chrono::seconds(10));

  return 0;
}
