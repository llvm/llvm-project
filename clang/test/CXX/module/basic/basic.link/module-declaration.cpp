// Tests for module-declaration syntax.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fmodule-file=x=%t/x.pcm %t/x.y.cppm -o %t/x.y.pcm
//
// Module implementation for unknown and known module. (The former is ill-formed.)
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/M.cpp \
// RUN:            -DTEST=1
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x=%t/x.pcm -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/M.cpp \
// RUN:            -DTEST=2
//
// Module interface for unknown and known module. (The latter is ill-formed due to
// redefinition.)
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=3
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=4
//
// Miscellaneous syntax.
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=7
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=8
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=9
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M.cpp \
// RUN:            -DTEST=10

//--- x.cppm
export module x;
int a, b;

//--- x.y.cppm
export module x.y;
int c;

//--- M.cpp

#if TEST == 1
module z; // expected-error {{module 'z' not found}}
#elif TEST == 2
module x; // expected-no-diagnostics
#elif TEST == 3
export module z; // expected-no-diagnostics
#elif TEST == 4
export module x; // expected-no-diagnostics
#elif TEST == 7
export module z elderberry; // expected-error {{expected ';'}} expected-error {{a type specifier is required}}
#elif TEST == 9
export module z [[fancy]]; // expected-warning {{unknown attribute 'fancy' ignored}}
#elif TEST == 10
export module z [[maybe_unused]]; // expected-error-re {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
#else
// expected-no-diagnostics
#endif
