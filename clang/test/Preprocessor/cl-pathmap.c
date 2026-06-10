// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// REQUIRES: x86-registered-target
// RUN: %clang_cl --target=x86_64-windows-msvc -E /pathmap:%p=x:/path-to/ -- %s | FileCheck %s --check-prefix CHECK-REPRODUCABLE
// CHECK-REPRODUCABLE: filename: "x:\\path-to\\cl-pathmap.c"
// CHECK-REPRODUCABLE-NOT: filename:

//NOTE: currently . and .\ just gets eliminated after mapping 
// RUN: %clang_cl -E /pathmap:%p=. %s | FileCheck %s --check-prefix CHECK-SIMPLE
// CHECK-SIMPLE: filename: "cl-pathmap.c"
// CHECK-SIMPLE-NOT: filename:
filename: __FILE__
