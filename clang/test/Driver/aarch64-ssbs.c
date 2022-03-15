// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8a+ssbs   %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -mcpu=cortex-x1      %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -mcpu=cortex-x1c     %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -mcpu=cortex-a77     %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+ssbs"

// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8a+nossbs %s 2>&1 | FileCheck %s --check-prefix=NOSSBS
// RUN: %clang -### -target aarch64-none-none-eabi -mcpu=cortex-x1c+nossbs %s 2>&1 | FileCheck %s --check-prefix=NOSSBS
// NOSSBS: "-target-feature" "-ssbs"

// RUN: %clang -### -target aarch64-none-none-eabi                      %s 2>&1 | FileCheck %s --check-prefix=ABSENTSSBS
// ABSENTSSBS-NOT: "-target-feature" "+ssbs"
// ABSENTSSBS-NOT: "-target-feature" "-ssbs"
