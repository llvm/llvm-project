// RUN: %clang -### -S --target=riscv64 %s 2>&1 | FileCheck %s
// RUN: %clang -### -S --target=riscv64 -msmall-data-limit=8 %s 2>&1 | FileCheck %s --check-prefix=EIGHT

// CHECK-NOT: "-msmall-data-limit"
// EIGHT: "-msmall-data-limit" "8"
