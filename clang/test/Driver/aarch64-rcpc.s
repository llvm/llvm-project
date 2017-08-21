// RUN: %clang -### -target aarch64 %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target aarch64 -march=armv8.1a %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target aarch64 -march=armv8.2a %s 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// CHECK-NONE-NOT: "-target-feature" "+rcpc"

// RUN: %clang -### -target aarch64 -march=armv8.2a+rcpc %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -march=armv8.3a+rcpc %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a75 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a55 %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+rcpc"

// RUN: %clang -### -target aarch64 -mcpu=cortex-a75+norcpc %s 2>&1 | FileCheck --check-prefix=CHECK-NO-RCPC %s
// RUN: %clang -### -target aarch64 -mcpu=cortex-a55+norcpc %s 2>&1 | FileCheck --check-prefix=CHECK-NO-RCPC %s
// CHECK-NO-RCPC: "-rcpc"
