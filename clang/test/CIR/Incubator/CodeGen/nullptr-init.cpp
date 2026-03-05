// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir -check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=LLVM %s

void t1() {
  int *p1 = nullptr;
  int *p2 = 0;
  int *p3 = (int*)0;
}

// CIR:      cir.func {{.*}} @_Z2t1v()
// CIR-NEXT:     %[[P1:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p1", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[P2:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p2", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[P3:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p3", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[NULLPTR1:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR1]], %[[P1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[NULLPTR2:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR2]], %[[P2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[NULLPTR3:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR3]], %[[P3]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     cir.return
// CIR-NEXT: }

// LLVM:      define{{.*}} @_Z2t1v()
// LLVM-NEXT:     %[[P1:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     %[[P2:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     %[[P3:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     store ptr null, ptr %[[P1]], align 8
// LLVM-NEXT:     store ptr null, ptr %[[P2]], align 8
// LLVM-NEXT:     store ptr null, ptr %[[P3]], align 8
// LLVM-NEXT:     ret void
// LLVM-NEXT: }

// Verify that we're capturing side effects during null pointer initialization.
int t2() {
  int x = 0;
  int *p = (x = 1, nullptr);
  return x;
}

// Note: An extra null pointer constant gets emitted as a result of visiting the
//       compound initialization expression. We could avoid this by capturing
//       the result of the compound initialization expression and explicitly
//       casting it to the required type, but a redundant constant seems less
//       intrusive than a redundant bitcast.

// CIR:       cir.func {{.*}} @_Z2t2v()
// CIR-NEXT:      %[[RETVAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:      %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT:      %[[P:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init] {alignment = 8 : i64}
// CIR-NEXT:      %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:      cir.store{{.*}} %[[ZERO]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:      cir.store{{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      %[[NULLPTR_EXTRA:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-NEXT:      %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:      cir.store{{.*}} %[[NULLPTR]], %[[P]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:      %[[X_VAL:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.store{{.*}} %[[X_VAL]], %[[RETVAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      %[[RETVAL:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.return %[[RETVAL]] : !s32i
// CIR-NEXT:  }

// LLVM:      define{{.*}} @_Z2t2v()
// LLVM-NEXT:     %[[RETVAL_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:     %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:     %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     store i32 0, ptr %[[X]], align 4
// LLVM-NEXT:     store i32 1, ptr %[[X]], align 4
// LLVM-NEXT:     store ptr null, ptr %[[P]], align 8
// LLVM-NEXT:     %[[X_VAL:.*]] = load i32, ptr %[[X]], align 4
// LLVM-NEXT:     store i32 %[[X_VAL]], ptr %[[RETVAL_ADDR]], align 4
// LLVM-NEXT:     %[[RETVAL:.*]] = load i32, ptr %[[RETVAL_ADDR]], align 4
// LLVM-NEXT:     ret i32 %[[RETVAL]]
// LLVM-NEXT: }
