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
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/M1.cpp
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x=%t/x.pcm -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/M2.cpp
//
// Module interface for unknown and known module. (The latter is ill-formed due to
// redefinition.)
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M3.cpp
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M4.cpp
//
// Miscellaneous syntax.
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M5.cpp
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M6.cpp
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M7.cpp
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/M8.cpp

//--- x.cppm
export module x;
int a, b;

//--- x.y.cppm
export module x.y;
int c;

//--- M1.cpp
module z; // expected-error {{module 'z' not found}}

//--- M2.cpp
module x; // expected-no-diagnostics

//--- M3.cpp
export module z; // expected-no-diagnostics

//--- M4.cpp
export module x; // expected-no-diagnostics

//--- M5.cpp
export module z elderberry; // expected-error {{expected ';'}} expected-error {{a type specifier is required}}

//--- M6.cpp
export module z [[]]; // expected-no-diagnostics

//--- M7.cpp
export module z [[fancy]]; // expected-warning {{unknown attribute 'fancy' ignored}}

//--- M8.cpp
export module z [[maybe_unused]]; // expected-error-re {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
