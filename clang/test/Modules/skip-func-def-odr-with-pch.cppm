// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// Testing the behavior of `-fskip-odr-check-in-gmf`
// RUN: %clang_cc1 -std=c++20 -DDIFFERENT -fskip-odr-check-in-gmf %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf -x c++-header %t/foo.h -emit-pch -o %t/foo.pch
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf -include-pch %t/foo.pch %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf -fprebuilt-module-path=%t  \
// RUN:    %t/C.cpp -verify -fsyntax-only

//--- foo.h
#ifndef FOO_H
#define FOO_H
class A {
public:
#ifndef DIFFERENT
  void func() {

  }
#endif
};
#endif

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::A;

//--- B.cppm
module;
#include "foo.h"
export module B;
export using ::A;

//--- C.cpp
import A;
import B;
// expected-no-diagnostics
void C() {
  A a;
  a.func();
}
