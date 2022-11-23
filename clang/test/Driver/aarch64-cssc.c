// Test that target feature cssc is implemented and available correctly
// RUN: %clang -### -target aarch64-none-none-eabi                         %s 2>&1 | FileCheck %s --check-prefix=ABSENT_CSSC
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.9-a        %s 2>&1 | FileCheck %s --check-prefix=ABSENT_CSSC
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.8-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.9-a+nocssc %s 2>&1 | FileCheck %s --check-prefix=NO_CSSC
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.4-a        %s 2>&1 | FileCheck %s --check-prefix=ABSENT_CSSC
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.3-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv9.4-a+nocssc %s 2>&1 | FileCheck %s --check-prefix=NO_CSSC

// CHECK: "-target-feature" "+cssc"
// NO_CSSC: "-target-feature" "-cssc"
// ABSENT_CSSC-NOT: "-target-feature" "+cssc"
// ABSENT_CSSC-NOT: "-target-feature" "-cssc"
