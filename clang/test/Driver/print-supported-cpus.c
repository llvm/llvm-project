// Test that --print-supported-cpus lists supported CPU models.

// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: aarch64-registered-target

// RUN: %clang --target=x86_64-unknown-linux-gnu --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-X86

// Test -mcpu=help and -mtune=help alises.
// RUN: %clang --target=x86_64-unknown-linux-gnu -mcpu=help 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-X86

// RUN: %clang --target=x86_64-unknown-linux-gnu -mtune=help -fuse-ld=dummy 2>&1 | \
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

// RUN: %clang --target=arm64-apple-macosx --print-supported-cpus 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-AARCH64 --implicit-check-not=apple-latest

// CHECK-AARCH64: Target: arm64-apple-macosx
// CHECK-AARCH64: apple-m1
// CHECK-AARCH64: apple-m2
// CHECK-AARCH64: apple-m3
// CHECK-AARCH64: Use -mcpu or -mtune to specify the target's processor.
