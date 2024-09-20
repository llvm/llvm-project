// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:   -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fsyntax-only -verify \
// RUN:   -fprebuilt-module-path=%t

//--- foo.h
inline auto x = []{};

//--- a.cppm
module;
#include "foo.h"
export module a;
export using ::x;

//--- b.cppm
module;
import a;
#include "foo.h"
export module b;
export using ::x;

//--- test.cc
// expected-no-diagnostics
import a;
import b;
void test() {
  x();
}
