// RUN: %clang -### -fpreferred-function-alignment=16 %s 2>&1 | FileCheck %s -check-prefix CHECK-16
// RUN: not %clang -### -fpreferred-function-alignment=3 %s 2>&1 | FileCheck %s -check-prefix CHECK-INVALID

// CHECK-16: "-preferred-function-alignment" "4"
// CHECK-INVALID: invalid integral value '3' in '-fpreferred-function-alignment=3'
