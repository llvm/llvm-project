// Test that -print-target-triple prints correct triple.

// RUN: %clang -print-target-triple 2>&1 \
// RUN:     --target=x86_64-linux-gnu \
// RUN:   | FileCheck %s
// CHECK: x86_64-unknown-linux-gnu
