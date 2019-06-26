// Test that the --print-supported-cpus flag works
// Also test its aliases: -mcpu=? and -mtune=?

// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target

// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN:   --print-supported-cpus 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-X86
// CHECK-X86: Target: x86_64-unknown-linux-gnu
// CHECK-X86: corei7
// CHECK-X86: Use -mcpu or -mtune to specify the target's processor.

// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN:   -mcpu=? 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-X86-MCPU
// CHECK-X86-MCPU: Target: x86_64-unknown-linux-gnu
// CHECK-X86-MCPU: corei7
// CHECK-X86-MCPU: Use -mcpu or -mtune to specify the target's processor.

// RUN: %clang --target=arm-unknown-linux-android \
// RUN:   --print-supported-cpus 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ARM
// CHECK-ARM: Target: arm-unknown-linux-android
// CHECK-ARM: cortex-a73
// CHECK-ARM: cortex-a75
// CHECK-ARM: Use -mcpu or -mtune to specify the target's processor.

// RUN: %clang --target=arm-unknown-linux-android \
// RUN:   -mtune=? 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ARM-MTUNE
// CHECK-ARM-MTUNE: Target: arm-unknown-linux-android
// CHECK-ARM-MTUNE: cortex-a73
// CHECK-ARM-MTUNE: cortex-a75
// CHECK-ARM-MTUNE: Use -mcpu or -mtune to specify the target's processor.
