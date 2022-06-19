// Test that target feature hbc is implemented and available correctly
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.7-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a       %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+nohbc %s 2>&1 | FileCheck %s --check-prefix=NO_HBC
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.2-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a       %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+nohbc %s 2>&1 | FileCheck %s --check-prefix=NO_HBC

// CHECK: "-target-feature" "+hbc"
// NO_HBC: "-target-feature" "-hbc"
