// RUN: %clang -target nios2--- %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target nios2--- -mcpu=r1 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R1 %s
// RUN: %clang -target nios2--- -mcpu=nios2r1 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R1 %s
// RUN: %clang -target nios2--- -march=r1 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R1 %s
// RUN: %clang -target nios2--- -march=nios2r1 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R1 %s

// RUN: %clang -target nios2--- -mcpu=r2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R2 %s
// RUN: %clang -target nios2--- -mcpu=nios2r2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R2 %s
// RUN: %clang -target nios2--- -march=r2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R2 %s
// RUN: %clang -target nios2--- -march=nios2r2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-R2 %s

// CHECK: "-triple" "nios2---"
// CHECK-R1: "-triple" "nios2---"
// CHECK-R1: "-target-cpu" "nios2r1"
// CHECK-R2: "-triple" "nios2---"
// CHECK-R2: "-target-cpu" "nios2r2"
