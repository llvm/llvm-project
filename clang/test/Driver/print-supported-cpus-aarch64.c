// Test that --print-supported-cpus lists supported CPU models, including aliases.

// REQUIRES: aarch64-registered-target

// RUN: %clang --target=arm64-apple-macosx --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK --implicit-check-not=apple-latest

// CHECK: Target: arm64-apple-macosx

// CHECK: apple-a11
// CHECK: apple-a12
// CHECK: apple-a13
// CHECK: apple-a14
// CHECK: apple-a15
// CHECK: apple-a16
// CHECK: apple-a17
// CHECK: apple-a7
// CHECK: apple-a8
// CHECK: apple-a9
// CHECK: apple-m1
// CHECK: apple-m2
// CHECK: apple-m3
// CHECK: apple-m4
// CHECK: apple-s4
// CHECK: apple-s5

// CHECK: Use -mcpu or -mtune to specify the target's processor.
