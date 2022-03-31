// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++03 -pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof %s 2>&1 | FileCheck %s
// rdar://9133072

// In C++11 mode, this is allowed, so don't warn in pedantic mode.
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -Wnewline-eof -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -Werror -pedantic %s

// Make sure the diagnostic shows up properly at the end of the last line.
// CHECK: newline-eof.c:[[@LINE+3]]:67

// The following line isn't terminated, don't fix it.
void foo(void) {} // expected-warning{{no newline at end of file}}