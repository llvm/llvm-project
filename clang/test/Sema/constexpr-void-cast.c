// RUN: %clang_cc1 -x c -fsyntax-only %s -verify=c -std=c11
// RUN: %clang_cc1 -x c -fsyntax-only %s -pedantic -verify=c-pedantic -std=c11
// RUN: %clang_cc1 -x c -fsyntax-only %s -verify=c -std=c11 -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -x c -fsyntax-only %s -pedantic -verify=c-pedantic -std=c11 -fexperimental-new-constant-interpreter
//
// RUN: %clang_cc1 -x c++ -fsyntax-only %s -verify=cxx
// RUN: %clang_cc1 -x c++ -fsyntax-only %s -pedantic -verify=cxx-pedantic
// RUN: %clang_cc1 -x c++ -fsyntax-only %s -verify=cxx -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -x c++ -fsyntax-only %s -pedantic -verify=cxx-pedantic -fexperimental-new-constant-interpreter

// c-no-diagnostics
// cxx-no-diagnostics

void f(void);
struct S {char c;} s;
_Static_assert(&s != (void *)&f, ""); // c-pedantic-warning {{not an integer constant expression}} \
                                      // c-pedantic-note {{this conversion is not allowed in a constant expression}} \
                                      // cxx-pedantic-warning {{'_Static_assert' is a C11 extension}}
