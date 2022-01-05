// RUN: %clang -### -target arm-none-none-eabi -march=armv8.4a+pmuv3p4 %s 2>&1 | FileCheck --check-prefix=CHECK-PERFMON %s
// RUN: %clang -### -target arm-none-none-eabi -march=armv8.2a+pmuv3p4 %s 2>&1 | FileCheck --check-prefix=CHECK-PERFMON %s
// CHECK-PERFMON: "-target-feature" "+perfmon"

// RUN: %clang -### -target arm-none-none-eabi -march=armv8.4a+nopmuv3p4 %s 2>&1 | FileCheck --check-prefix=CHECK-NOPERFMON %s
// RUN: %clang -### -target arm-none-none-eabi -march=armv8.2a+nopmuv3p4 %s 2>&1 | FileCheck --check-prefix=CHECK-NOPERFMON %s
// CHECK-NOPERFMON: "-target-feature" "-perfmon"

// RUN: %clang -### -target arm-none-none-eabi                 %s 2>&1 | FileCheck %s --check-prefix=ABSENTPERFMON
// RUN: %clang -### -target arm-none-none-eabi -march=armv8.4a %s 2>&1 | FileCheck %s --check-prefix=ABSENTPERFMON
// RUN: %clang -### -target arm-none-none-eabi -march=armv8.2a %s 2>&1 | FileCheck %s --check-prefix=ABSENTPERFMON
// ABSENTPERFMON-NOT: "-target-feature" "+perfmon"
// ABSENTPERFMON-NOT: "-target-feature" "-perfmon"