// RUN: %clang -S %s -o - | FileCheck %s
.macro  my_macro, trace=1, uaccess=1
.if \uaccess
// CHECK: .if \uaccess
// CHECK-NOT: .if 곎ss
// CHECK: my_macro trace=1
        my_macro trace=1
.endif
.endm

foo:
        my_macro trace=0
