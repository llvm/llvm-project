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
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/z_impl.cppm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x=%t/x.pcm -fmodule-file=x.y=%t/x.y.pcm -verify -x c++ %t/x_impl.cppm
//
// Module interface for unknown and known module. (The latter is ill-formed due to
// redefinition.)
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/z_interface.cppm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/x_interface.cppm
//
// Miscellaneous syntax.
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/invalid_module_name.cppm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/empty_attribute.cppm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/fancy_attribute.cppm
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x.y=%t/x.y.pcm -verify %t/maybe_unused_attribute.cppm

//--- x.cppm
export module x;
int a, b;

//--- x.y.cppm
export module x.y;
int c;

//--- z_impl.cppm
module z; // expected-error {{module 'z' not found}}

//--- x_impl.cppm
// expected-no-diagnostics
module x;

//--- z_interface.cppm
// expected-no-diagnostics
export module z;

//--- x_interface.cppm
// expected-no-diagnostics
export module x;

//--- invalid_module_name.cppm
export module z elderberry;
// expected-error@-1 {{unexpected preprocessing token 'elderberry' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- empty_attribute.cppm
// expected-no-diagnostics
export module z [[]];

//--- fancy_attribute.cppm
export module z [[fancy]]; // expected-warning {{unknown attribute 'fancy' ignored}}

//--- maybe_unused_attribute.cppm
export module z [[maybe_unused]]; // expected-error-re {{'maybe_unused' attribute cannot be applied to a module{{$}}}}
