// Test that we will skip ODR checks for declarations from PCH if they
// were from GMF.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.cppm \
// RUN:   -o %t/A.pcm -fskip-odr-check-in-gmf
// RUN: %clang_cc1 -std=c++20 -DDIFF -x c++-header %t/foo.h \
// RUN:   -emit-pch -o %t/foo.pch -fskip-odr-check-in-gmf
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fmodule-file=A=%t/A.pcm -include-pch \
// RUN:   %t/foo.pch -verify -fsyntax-only -fskip-odr-check-in-gmf

//--- foo.h
#ifndef FOO_H
#define FOO_H
inline int foo() {
#ifndef DIFF
  return 43;
#else
  return 45;
#endif
}

class f {
public:
  int mem() {
#ifndef DIFF
    return 47;
#else
    return 45;
#endif
  }
};
#endif

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::foo;
export using ::f;

//--- B.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module B;
import A;
export int b = foo() + f().mem();
