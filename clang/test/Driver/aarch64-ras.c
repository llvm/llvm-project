// RUN: %clang -target aarch64-none-none-eabi -march=armv8a+ras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang -target aarch64-none-none-eabi -mcpu=generic+ras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang -target aarch64-none-none-eabi -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang -target aarch64-none-none-eabi -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// CHECK-RAS: "-target-feature" "+ras"

// RUN: %clang -target aarch64-none-none-eabi -march=armv8a+noras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORAS %s
// RUN: %clang -target aarch64-none-none-eabi -mcpu=generic+noras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORAS %s
// CHECK-NORAS: "-target-feature" "-ras"
