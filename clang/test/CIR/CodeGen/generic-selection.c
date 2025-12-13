// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  int a;
  int r = _Generic(a, double: 1, float: 2, int: 3, default: 4);
}

// CIR: %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR: %[[RES:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init]
// CIR: %[[RES_VAL:.*]] = cir.const #cir.int<3> : !s32i
// CIR: cir.store{{.*}} %[[RES_VAL]], %[[RES]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RES:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 3, ptr %[[RES]], align 4

// OGCG: %[[A:.*]] = alloca i32, align 4
// OGCG: %[[RES:.*]] = alloca i32, align 4
// OGCG: store i32 3, ptr %[[RES]], align 4
