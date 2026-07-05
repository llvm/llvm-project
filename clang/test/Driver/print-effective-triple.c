// Test that -print-target-triple prints correct triple.

// RUN: %clang -print-effective-triple \
// RUN:     --target=thumb-linux-gnu 2>&1 \
// RUN:   | FileCheck %s
// CHECK: armv4t-unknown-linux-gnu
