// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void gnu_null_expr() {
  long a = __null;
  int *b = __null;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["b", init]
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s64i
// CIR: cir.store {{.*}} %[[CONST_0]], %[[A_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR: %[[CONST_NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR: cir.store {{.*}} %[[CONST_NULL]], %[[B_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store i64 0, ptr %[[A_ADDR]], align 8
// LLVM: store ptr null, ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[B_ADDR:.*]] = alloca ptr, align 8
// OGCG: store i64 0, ptr %[[A_ADDR]], align 8
// OGCG: store ptr null, ptr %[[B_ADDR]], align 8
