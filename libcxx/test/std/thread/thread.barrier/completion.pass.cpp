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

// This test requires the dylib support introduced in D68480,
// which hasn't shipped yet.
// XFAIL: use_system_cxx_lib && x86_64-apple
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.15
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.14
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

// <barrier>

#include <barrier>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  int x = 0;
  auto comp = [&]() { x += 1; };
  std::barrier<decltype(comp)> b(2, comp);

  std::thread t = support::make_test_thread([&](){
      for(int i = 0; i < 10; ++i)
        b.arrive_and_wait();
  });

  for(int i = 0; i < 10; ++i)
    b.arrive_and_wait();

  assert(x == 10);
  t.join();
  return 0;
}
