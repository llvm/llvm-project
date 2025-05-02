// RUN: %clang_cc1 -fsyntax-only -verify -Wduplicate-decl-specifier %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-duplicate-decl-specifier -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wc++-compat -Wno-duplicate-decl-specifier %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wno-duplicate-decl-specifier %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// good-no-diagnostics

// Note: we treat this as a warning in C++, so you get the same diagnostics in
// either language mode. However, GCC diagnoses this as an error, so the
// compatibility warning has value.
const const int i = 12; // expected-warning {{duplicate 'const' declaration specifier}}

__attribute__((address_space(1)))
__attribute__((address_space(1))) // expected-warning {{multiple identical address spaces specified for type}}
int j = 12;

