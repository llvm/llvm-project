// Check that -chain-include works with relative paths, and with paths whose
// separators differ from the native ones (relevant on Windows).

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: cd %t && %clang_cc1 -chain-include ./first.h -chain-include second.h \
// RUN:   -fsyntax-only -verify tu.c
// RUN: %clang_cc1 -chain-include %t/first.h -chain-include %t/second.h \
// RUN:   -fsyntax-only -verify %t/tu.c

//--- first.h
int a;

//--- second.h
int b;

//--- tu.c
// expected-no-diagnostics
int f(void) { return a + b; }
