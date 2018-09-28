// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.4a+rng %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.5a+rng %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+rand"

// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.4a+norng %s 2>&1 | FileCheck %s --check-prefix=NORAND
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.5a+norng %s 2>&1 | FileCheck %s --check-prefix=NORAND
// NORAND: "-target-feature" "-rand"

// RUN: %clang -### -target aarch64-none-none-eabi                 %s 2>&1 | FileCheck %s --check-prefix=ABSENTRAND
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.4a %s 2>&1 | FileCheck %s --check-prefix=ABSENTRAND
// RUN: %clang -### -target aarch64-none-none-eabi -march=armv8.5a %s 2>&1 | FileCheck %s --check-prefix=ABSENTRAND
// ABSENTRAND-NOT: "-target-feature" "+rand"
// ABSENTRAND-NOT: "-target-feature" "-rand"
