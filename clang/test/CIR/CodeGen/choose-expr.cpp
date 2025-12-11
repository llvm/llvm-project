// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void choose_expr() {
  int a = __builtin_choose_expr(1, 2, 3);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store {{.*}} %[[CONST_2]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 2, ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 2, ptr %[[A_ADDR]], align 4
