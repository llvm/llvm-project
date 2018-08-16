// Test that -print-target-triple prints correct triple.

// RUN: %clang -print-effective-triple 2>&1 \
// RUN:     --target=thumb-linux-gnu \
// RUN:   | FileCheck %s
// CHECK: armv4t-unknown-linux-gnu
