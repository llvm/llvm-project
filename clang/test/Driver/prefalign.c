// RUN: %clang -### -fpreferred-function-alignment=16 %s 2>&1 | FileCheck %s -check-prefix CHECK-16
// RUN: not %clang -### -fpreferred-function-alignment=131072 %s 2>&1 | FileCheck %s -check-prefix CHECK-131072
// RUN: not %clang -### -fpreferred-function-alignment=3 %s 2>&1 | FileCheck %s -check-prefix CHECK-3

// CHECK-16: "-fpreferred-function-alignment=16"
// CHECK-131072: invalid integral value '131072' in '-fpreferred-function-alignment=131072'
// CHECK-3: alignment is not a power of 2 in '-fpreferred-function-alignment=3'
