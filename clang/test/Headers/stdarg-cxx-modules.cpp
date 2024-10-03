// RUN: rm -fR %t
// RUN: split-file %s %t

/// FIXME: Using absolute path and explicitly specifying output files since
/// downstream branch has module file bypass for inode reuse problem on linux file systems.
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/h1.h -o %t/h1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/h2.h -fmodule-file=%t/h1.pcm -o %t/h2.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/main.cpp -fmodule-file=%t/h1.pcm -fmodule-file=%t/h2.pcm

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
