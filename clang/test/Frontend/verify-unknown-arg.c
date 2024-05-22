// RUN: not %clang_cc1 -asdf -verify %s 2>&1 | FileCheck %s

// expected-no-diagnostics

//      CHECK: error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT: (frontend): unknown argument: '-asdf'
