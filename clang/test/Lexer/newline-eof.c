// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof %s 2>&1 | FileCheck %s
// rdar://9133072

// Make sure the diagnostic shows up properly at the end of the last line.
// CHECK: newline-eof.c:9:63

// The following line isn't terminated, don't fix it.
void foo() {} // expected-warning{{no newline at end of file}}