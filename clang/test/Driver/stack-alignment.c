// RUN: not %clang -### -mstack-alignment=-1 %s 2>&1 | FileCheck %s --check-prefix=CHECK_NEG_1
// RUN: %clang -### -mstack-alignment=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK_0
// RUN: %clang -### -mstack-alignment=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK_1
// RUN: %clang -### -mstack-alignment=4 %s 2>&1 | FileCheck %s --check-prefix=CHECK_4
// RUN: not %clang -### -mstack-alignment=5 %s 2>&1 | FileCheck %s --check-prefix=CHECK_5

// CHECK_NEG_1: error: invalid argument '-1' to -mstack-alignment=
// CHECK_0: -mstack-alignment=0
// CHECK_1: -mstack-alignment=1
// CHECK_4: -mstack-alignment=4
// CHECK_5: error: alignment is not a power of 2 in '-mstack-alignment=5'
