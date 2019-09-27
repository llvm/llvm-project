// RUN: not %clang -verify %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

// Test that -verify is strictly rejected as unknown by the driver.
// CHECK: unknown argument: '-verify'
