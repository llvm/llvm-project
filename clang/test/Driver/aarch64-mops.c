// Test that target feature mops is implemented and available correctly
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+mops %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+mops"

// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+nomops %s 2>&1 | FileCheck %s --check-prefix=NO_MOPS
// NO_MOPS: "-target-feature" "-mops"
