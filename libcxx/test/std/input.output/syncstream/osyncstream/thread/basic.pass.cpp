//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_osyncstream;

// Basic test whether the code works in a threaded environment.
// Using timing the output order should be stable.
// several_threads.pass.cpp tests with more threads.

#include <syncstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <cassert>
#include <iostream>

#include "test_macros.h"

static std::basic_ostringstream<char> ss;
static const char a = 'a';
static const char b = 'b';
static const char c = 'c';
static const char d = 'd';

void f1() {
  std::basic_osyncstream<char> out(ss);
  out << a;
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  out << b;
}

void f2() {
  std::basic_osyncstream<char> out(ss);
  out << c;
  out << d;
}

int main(int, char**) {
  std::thread t1(f1);
  std::thread t2(f2);
  t1.join();
  t2.join();

  assert(ss.str() == "cdab");

  return 0;
}
