// Test that the --print-supported-cpus flag works

// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN:   --print-supported-cpus 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-X86
// CHECK-X86: Target: x86_64-unknown-linux-gnu
// CHECK-X86: corei7
// CHECK-X86: Use -mcpu or -mtune to specify the target's processor.

// REQUIRES: arm-registered-target
// RUN: %clang --target=arm-unknown-linux-android \
// RUN:   --print-supported-cpus 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ARM
// CHECK-ARM: Target: arm-unknown-linux-android
// CHECK-ARM: cortex-a73
// CHECK-ARM: cortex-a75
// CHECK-ARM: Use -mcpu or -mtune to specify the target's processor.
