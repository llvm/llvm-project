// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int _Complex ci;

float _Complex cf;

int _Complex ci2 = { 1, 2 };

float _Complex cf2 = { 1.0f, 2.0f };

// CIR: cir.global external @ci = #cir.zero : !cir.complex<!s32i>
// CIR: cir.global external @cf = #cir.zero : !cir.complex<!cir.float>
// CIR: cir.global external @ci2 = #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.global external @cf2 = #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>

// LLVM: {{.*}} = global { i32, i32 } zeroinitializer, align 4
// LLVM: {{.*}} = global { float, float } zeroinitializer, align 4
// LLVM: {{.*}} = global { i32, i32 } { i32 1, i32 2 }, align 4
// LLVM: {{.*}} = global { float, float } { float 1.000000e+00, float 2.000000e+00 }, align 4

// OGCG: {{.*}} = global { i32, i32 } zeroinitializer, align 4
// OGCG: {{.*}} = global { float, float } zeroinitializer, align 4
// OGCG: {{.*}} = global { i32, i32 } { i32 1, i32 2 }, align 4
// OGCG: {{.*}} = global { float, float } { float 1.000000e+00, float 2.000000e+00 }, align 4
