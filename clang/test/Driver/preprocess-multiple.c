// RUN: %clang -E %s %s | FileCheck %s
// Test that the driver can preprocess multiple files.

X
// CHECK: X
// CHECK: X
