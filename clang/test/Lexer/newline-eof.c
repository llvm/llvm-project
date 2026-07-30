// Allowing a file to end without a newline was adopted as a Defect Report in
// WG21 (CWG787) and in WG14 (added to the list of changes which apply to
// earlier revisions of C in C2y). So it should not issue a pedantic diagnostic
// in any language mode.

// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof -verify %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -std=c89 -pedantic -Wno-comment -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++03 -pedantic -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -Wnewline-eof -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=c++11 -Werror -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof %s 2>&1 | FileCheck %s
// good-no-diagnostics

// Make sure the diagnostic shows up properly at the end of the last line.
// CHECK: newline-eof.c:[[@LINE+3]]:67

// The following line isn't terminated, don't fix it.
void foo(void) {} // expected-warning{{no newline at end of file}}