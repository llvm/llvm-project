// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fprebuilt-module-path=%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -fprebuilt-module-path=%t -verify %t/C.cpp

//--- foo.h
struct Bar {};
extern "C" void foo(struct Bar);

//--- A.cppm
module;
#include "foo.h"
export module A;
export extern "C" using ::foo;
//--- B.cppm
module;
import A;
export module B;

//--- C.cpp
// expected-no-diagnostics
import B;
#include "foo.h"
void test() {
  foo(Bar());
}
