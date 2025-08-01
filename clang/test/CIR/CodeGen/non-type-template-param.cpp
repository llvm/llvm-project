// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

template <const int N>
void template_foo() {
  int a = N + 5;
}

// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[ADD:.*]] = cir.binop(add, %[[CONST_1]], %[[CONST_2]]) nsw : !s32i
// CIR: cir.store{{.*}} %[[ADD]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 6, ptr %[[INIT]], align 4

// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: store i32 6, ptr %[[INIT]], align 4

void foo() {
  template_foo<1>();
}
