// RUN: %clang_analyze_cc1 -analyzer-checker=unix.StdCLibraryFunctions -verify %s
// RUN: %clang_analyze_cc1 -triple i686-unknown-linux -analyzer-checker=unix.StdCLibraryFunctions -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=unix.StdCLibraryFunctions -verify %s
// RUN: %clang_analyze_cc1 -triple armv7-a15-linux -analyzer-checker=unix.StdCLibraryFunctions -verify %s
// RUN: %clang_analyze_cc1 -triple thumbv7-a15-linux -analyzer-checker=unix.StdCLibraryFunctions -verify %s

// This test tests crashes that occur when standard functions are available
// for inlining.

// expected-no-diagnostics

int isdigit(int _) { return !0; }
void test_redefined_isdigit(int x) {
  int (*func)(int) = isdigit;
  for (; func(x);) // no-crash
    ;
}
