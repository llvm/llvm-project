// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: clang || apple-clang
// XFAIL: *

// This tests is meant to demonstrate an existing ABI bug between the
// C++03 and C++11 implementations of std::function. It is not a real test.

// RUN: %{cxx} -c %s -o %t.first.o %{flags} %{compile_flags} -std=c++03 -g
// RUN: %{cxx} -c %s -o %t.second.o -DWITH_MAIN %{flags} %{compile_flags} -g -std=c++11
// RUN: %{cxx} -o %t.exe %t.first.o %t.second.o %{flags} %{link_flags} -g
// RUN: %{run}

#include <functional>
#include <cassert>

typedef std::function<void(int)> Func;

Func CreateFunc();

#ifndef WITH_MAIN
// In C++03, the functions call operator, which is a part of the vtable,
// is defined as 'void operator()(int)', but in C++11 it's
// void operator()(int&&)'. So when the C++03 version is passed to C++11 code
// the value of the integer is interpreted as its address.
void test(int x) {
  assert(x == 42);
}
Func CreateFunc() {
  Func f(&test);
  return f;
}
#else
int main(int, char**) {
  Func f = CreateFunc();
  f(42);
  return 0;
}
#endif
