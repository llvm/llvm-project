// RUN: %clang -### -target aarch64 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target aarch64 -march=armv8.1a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target aarch64 -march=armv8.2a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -target aarch64 -march=armv8.3a %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// CHECK-NONE-NOT: "-target-feature" "+dotprod"

// RUN: %clang -### -target aarch64 -march=armv8.2a+dotprod %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -march=armv8.3a+dotprod %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a75 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a76 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a55 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-r82 %s 2>&1 | FileCheck %s
// CHECK: "+dotprod"
