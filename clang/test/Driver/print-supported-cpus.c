// Test that --print-supported-cpus lists supported CPU models.

// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target

// RUN: %clang --target=x86_64-unknown-linux-gnu --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-X86

// Test -mcpu=? and -mtune=? alises.
// RUN: %clang --target=x86_64-unknown-linux-gnu -mcpu=? 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-X86

// RUN: %clang --target=x86_64-unknown-linux-gnu -mtune=? -fuse-ld=dummy 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-X86

// CHECK-NOT: warning: argument unused during compilation
// CHECK-X86: Target: x86_64-unknown-linux-gnu
// CHECK-X86: corei7
// CHECK-X86: Use -mcpu or -mtune to specify the target's processor.

// RUN: %clang --target=arm-unknown-linux-android --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ARM

// CHECK-ARM: Target: arm-unknown-linux-android
// CHECK-ARM: cortex-a73
// CHECK-ARM: cortex-a75
// CHECK-ARM: Use -mcpu or -mtune to specify the target's processor.
