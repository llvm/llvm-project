// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example1.cpp -D'DOT_BAR=.bar' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example2.cpp -D'MOD_ATTR=[[vendor::shiny_module]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example3.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example4.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example5.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example6.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext1.cpp -verify -E | FileCheck %t/cwg2947_ext1.cpp
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext3.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example1.cpp -D'DOT_BAR=.bar' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example2.cpp -D'MOD_ATTR=[[vendor::shiny_module]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example3.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example4.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example5.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_example6.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_ext1.cpp -verify -E | FileCheck %t/cwg2947_ext1.cpp
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_ext2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/cwg2947_ext3.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example1.cpp -D'DOT_BAR=.bar' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example2.cpp -D'MOD_ATTR=[[vendor::shiny_module]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example3.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example4.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example5.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_example6.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_ext1.cpp -verify -E | FileCheck %t/cwg2947_ext1.cpp
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_ext2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 %t/cwg2947_ext3.cpp -fsyntax-only -verify

//--- cwg2947_example1.cpp
// #define DOT_BAR .bar
export module foo DOT_BAR; // error: expansion of DOT_BAR; does not begin with ; or [
// expected-error@-1 {{unexpected preprocessing token '.' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- cwg2947_example2.cpp
export module M MOD_ATTR;        // OK
// expected-warning@-1 {{unknown attribute 'vendor::shiny_module' ignored}}

//--- cwg2947_example3.cpp
export module a
  .b;                         // error: preprocessing token after pp-module-name is not ; or [
// expected-error@-1 {{unexpected preprocessing token '.' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- cwg2947_example4.cpp
export module M [[
  attr1,
// expected-warning@-1 {{unknown attribute 'attr1' ignored}}
  attr2 ]] ;                 // OK
// expected-warning@-1 {{unknown attribute 'attr2' ignored}}

//--- cwg2947_example5.cpp
export module M
  [[ attr1,
// expected-warning@-1 {{unknown attribute 'attr1' ignored}}
  attr2 ]] ;                 // OK
// expected-warning@-1 {{unknown attribute 'attr2' ignored}}

//--- cwg2947_example6.cpp
export module M; int
// expected-warning@-1 {{extra tokens after semicolon in 'module' directive}}
  n;                         // OK

//--- cwg2947_ext1.cpp
// CHECK: export __preprocessed_module m; int x;
// CHECK: extern "C++" int *y = &x;
export module m; int x;
// expected-warning@-1 {{extra tokens after semicolon in 'module' directive}}
extern "C++" int *y = &x;

//--- cwg2947_ext2.cpp
export module x _Pragma("GCC warning \"Hi\"");
// expected-warning@-1 {{Hi}}

//--- cwg2947_ext3.cpp
export module x; _Pragma("GCC warning \"hi\""); // expected-warning {{hi}}
// expected-warning@-1 {{extra tokens after semicolon in 'module' directive}}
