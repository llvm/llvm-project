/* A Bison parser, made by GNU Bison 1.875.  */

// RUN: rm -rf %t.plist
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist -o %t.plist -verify %s
// RUN: FileCheck --input-file=%t.plist %s

// expected-no-diagnostics
int foo() {
  int *x = 0;
  return *x; // no-warning
}

// CHECK:   <key>diagnostics</key>
