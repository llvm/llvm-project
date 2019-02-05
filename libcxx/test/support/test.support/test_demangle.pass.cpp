// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "demangle.h"
#include <typeinfo>
#include <cassert>

struct MyType {};

template <class T, class U> struct ArgumentListID {};

int main(int, char**) {
  struct {
    const char* raw;
    const char* expect;
  } TestCases[] = {
      {typeid(int).name(), "int"},
      {typeid(MyType).name(), "MyType"},
      {typeid(ArgumentListID<int, MyType>).name(), "ArgumentListID<int, MyType>"}
  };
  const size_t size = sizeof(TestCases) / sizeof(TestCases[0]);
  for (size_t i=0; i < size; ++i) {
    const char* raw = TestCases[i].raw;
    const char* expect = TestCases[i].expect;
#ifdef TEST_HAS_NO_DEMANGLE
    assert(demangle(raw) == raw);
    ((void)expect);
#else
    assert(demangle(raw) == expect);
#endif
  }

  return 0;
}
