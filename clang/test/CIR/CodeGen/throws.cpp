// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  throw;
}

// CIR: cir.throw
// CIR: cir.unreachable

// LLVM: call void @__cxa_rethrow()
// LLVM: unreachable

// OGCG: call void @__cxa_rethrow()
// OGCG: unreachable

int foo1(int a, int b) {
  if (b == 0)
    throw;
  return a / b;
}

// CIR:  %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR:  %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:  %[[RES_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:  cir.store %{{.*}}, %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.store %{{.*}}, %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.scope {
// CIR:    %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:    %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:    %[[IS_B_ZERO:.*]] = cir.cmp(eq, %[[TMP_B]], %[[CONST_0]]) : !s32i, !cir.bool
// CIR:    cir.if %[[IS_B_ZERO]] {
// CIR:      cir.throw
// CIR:      cir.unreachable
// CIR:    }
// CIR:  }
// CIR:  %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[DIV_A_B:.*]] = cir.binop(div, %[[TMP_A:.*]], %[[TMP_B:.*]]) : !s32i
// CIR:  cir.store %[[DIV_A_B]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  %[[RESULT:.*]] = cir.load %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  cir.return %[[RESULT]] : !s32i

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RES_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 %{{.*}}, ptr %[[A_ADDR]], align 4
// LLVM: store i32 %{{.*}}, ptr %[[B_ADDR]], align 4
// LLVM: br label %[[CHECK_COND:.*]]
// LLVM: [[CHECK_COND]]:
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[IS_B_ZERO:.*]] = icmp eq i32 %[[TMP_B]], 0
// LLVM:  br i1 %[[IS_B_ZERO]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
// LLVM: [[IF_THEN]]:
// LLVM:  call void @__cxa_rethrow()
// LLVM:  unreachable
// LLVM: [[IF_ELSE]]:
// LLVM:  br label %[[IF_END:.*]]
// LLVM: [[IF_END]]:
// LLVM:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[DIV_A_B:.*]] = sdiv i32 %[[TMP_A]], %[[TMP_B]]
// LLVM:  store i32 %[[DIV_A_B]], ptr %[[RES_ADDR]], align 4
// LLVM:  %[[RESULT:.*]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:  ret i32 %[[RESULT]]

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 %{{.*}}, ptr %[[A_ADDR]], align 4
// OGCG: store i32 %{{.*}}, ptr %[[B_ADDR]], align 4
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG: %[[IS_B_ZERO:.*]] = icmp eq i32 %[[TMP_B]], 0
// OGCG: br i1 %[[IS_B_ZERO]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN]]:
// OGCG:  call void @__cxa_rethrow()
// OGCG:  unreachable
// OGCG: [[IF_END]]:
// OGCG:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG:  %[[DIV_A_B:.*]] = sdiv i32 %[[TMP_A]], %[[TMP_B]]
// OGCG:  ret i32 %[[DIV_A_B]]
