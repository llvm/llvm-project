// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

__attribute((aligned(32))) float a[128];
union {int a[4]; __attribute((aligned(32))) float b[4];} b;

// CIR: @a = #cir.zero {{.*}}alignment = 32
// CIR: @b = #cir.zero{{.*}}alignment = 32

// LLVM: @a = {{.*}}zeroinitializer, align 32
// LLVM: @b = {{.*}}zeroinitializer, align 32

// OGCG: @a = {{.*}}zeroinitializer, align 32
// OGCG: @b = {{.*}}zeroinitializer, align 32

long long int test5[1024];
// CIR: @test5 = #cir.zero {{.*}}alignment = 16
// LLVM: @test5 = {{.*}}global [1024 x i64] zeroinitializer, align 16
// OGCG: @test5 = {{.*}}global [1024 x i64] zeroinitializer, align 16

// TODO: Add more test cases from clang/test/CodeGen/alignment.c when we have
//       implemented compound literal expression support.
