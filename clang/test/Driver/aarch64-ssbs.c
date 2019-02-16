// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8a+ssbs   %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+ssbs"

// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8a+nossbs %s 2>&1 | FileCheck %s --check-prefix=NOSSBS
// NOSSBS: "-target-feature" "-ssbs"

// RUN: %clang -### -target aarch64-none-none-eabi                      %s 2>&1 | FileCheck %s --check-prefix=ABSENTSSBS
// ABSENTSSBS-NOT: "-target-feature" "+ssbs"
// ABSENTSSBS-NOT: "-target-feature" "-ssbs"
