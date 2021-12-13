// Test that target feature hbc is implemented and available correctly
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+hbc %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+hbc"

// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+nohbc %s 2>&1 | FileCheck %s --check-prefix=NO_HBC
// NO_HBC: "-target-feature" "-hbc"
