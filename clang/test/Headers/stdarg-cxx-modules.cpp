// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header h1.h
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header h2.h -fmodule-file=h1.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only main.cpp -fmodule-file=h1.pcm -fmodule-file=h2.pcm

//--- h1.h
#include <stdarg.h>
// expected-no-diagnostics

//--- h2.h
import "h1.h";
// expected-no-diagnostics

//--- main.cpp
import "h1.h";
import "h2.h";

void foo(int x, ...) {
  va_list v;
  va_start(v, x);
  va_end(v);
}
// expected-no-diagnostics
