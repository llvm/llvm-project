// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

int _Complex gci;

float _Complex gcf;

int _Complex gci2 = { 1, 2 };

float _Complex gcf2 = { 1.0f, 2.0f };

// CHECK: cir.global external {{.*}} = #cir.zero : !cir.complex<!s32i>
// CHECK: cir.global external {{.*}} = #cir.zero : !cir.complex<!cir.float>
// CHECK: cir.global external {{.*}} = #cir.complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CHECK: cir.global external {{.*}} = #cir.complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>

// LLVM: {{.*}} = global { i32, i32 } zeroinitializer, align 4
// LLVM: {{.*}} = global { float, float } zeroinitializer, align 4
// LLVM: {{.*}} = global { i32, i32 } { i32 1, i32 2 }, align 4
// LLVM: {{.*}} = global { float, float } { float 1.000000e+00, float 2.000000e+00 }, align 4