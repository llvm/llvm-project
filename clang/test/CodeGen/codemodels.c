// RUN: %clang_cc1 -emit-llvm  %s -o - | FileCheck %s -check-prefix=CHECK-NOMODEL
// RUN: %clang_cc1 -triple aarch64-unknown-none-eabi -emit-llvm -mcode-model tiny %s -o - | FileCheck %s -check-prefix=CHECK-TINY
// RUN: %clang_cc1 -emit-llvm -mcode-model small %s -o - | FileCheck %s -check-prefix=CHECK-SMALL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -mcode-model kernel %s -o - | FileCheck %s -check-prefix=CHECK-KERNEL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -mcode-model medium %s -o - | FileCheck %s -check-prefix=CHECK-MEDIUM
// RUN: %clang_cc1 -emit-llvm -mcode-model large %s -o - | FileCheck %s -check-prefix=CHECK-LARGE

// CHECK-TINY: !llvm.module.flags = !{{{.*}}}
// CHECK-TINY: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 0}
// CHECK-SMALL: !llvm.module.flags = !{{{.*}}}
// CHECK-SMALL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
// CHECK-KERNEL: !llvm.module.flags = !{{{.*}}}
// CHECK-KERNEL: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 2}
// CHECK-MEDIUM: !llvm.module.flags = !{{{.*}}}
// CHECK-MEDIUM: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 3}
// CHECK-LARGE: !llvm.module.flags = !{{{.*}}}
// CHECK-LARGE: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}
// CHECK-NOMODEL-NOT: Code Model
