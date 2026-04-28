// RUN: split-file %s %t

// 'main' implicitly returns 0 if it falls off the end of the function. CIR
// must initialize the return slot to 0 in the prologue so that loading it for
// the implicit return does not yield an undefined value.


//--- empty_body.c

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/empty_body.c -o %t/empty_body.cir
// RUN: FileCheck --input-file=%t/empty_body.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/empty_body.c -o %t/empty_body-cir.ll
// RUN: FileCheck --input-file=%t/empty_body-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/empty_body.c -o %t/empty_body.ll
// RUN: FileCheck --input-file=%t/empty_body.ll %s --check-prefix=OGCG

int main(void) {
}

// CIR-LABEL: cir.func{{.*}} @main
// CIR:         %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store %[[ZERO]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[LOAD:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.return %[[LOAD]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @main(
// LLVM:         %[[RET:.*]] = alloca i32
// LLVM:         store i32 0, ptr %[[RET]]
// LLVM:         %[[LOAD:.*]] = load i32, ptr %[[RET]]
// LLVM:         ret i32 %[[LOAD]]

// OGCG-LABEL: define{{.*}} i32 @main(
// OGCG:         ret i32 0


//--- nonempty_body.c

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/nonempty_body.c -o %t/nonempty_body.cir
// RUN: FileCheck --input-file=%t/nonempty_body.cir %s --check-prefix=CIR2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/nonempty_body.c -o %t/nonempty_body-cir.ll
// RUN: FileCheck --input-file=%t/nonempty_body-cir.ll %s --check-prefix=LLVM2
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/nonempty_body.c -o %t/nonempty_body.ll
// RUN: FileCheck --input-file=%t/nonempty_body.ll %s --check-prefix=OGCG2

int g;

int main(void) {
  if (g != 0)
    return g;
}

// CIR2-LABEL: cir.func{{.*}} @main
// CIR2:         %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR2:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR2:         cir.store %[[ZERO]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR2:         cir.if
// CIR2:           %[[G_ADDR:.*]] = cir.get_global @g
// CIR2:           %[[G:.*]] = cir.load{{.*}} %[[G_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR2:           cir.store %[[G]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR2:           %[[EXPL:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR2:           cir.return %[[EXPL]] : !s32i
// CIR2:         %[[IMPL:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR2:         cir.return %[[IMPL]] : !s32i

// LLVM2-LABEL: define{{.*}} i32 @main(
// LLVM2:         %[[RET:.*]] = alloca i32
// LLVM2:         store i32 0, ptr %[[RET]]
// LLVM2:         icmp ne
// LLVM2:         %[[G:.*]] = load i32, ptr @g
// LLVM2:         store i32 %[[G]], ptr %[[RET]]
// LLVM2:         %[[EXPL:.*]] = load i32, ptr %[[RET]]
// LLVM2:         ret i32 %[[EXPL]]
// LLVM2:         %[[IMPL:.*]] = load i32, ptr %[[RET]]
// LLVM2:         ret i32 %[[IMPL]]

// OGCG2-LABEL: define{{.*}} i32 @main(
// OGCG2:         %[[RET:.*]] = alloca i32
// OGCG2:         store i32 0, ptr %[[RET]]
// OGCG2:         icmp ne
// OGCG2:         %[[G:.*]] = load i32, ptr @g
// OGCG2:         store i32 %[[G]], ptr %[[RET]]
// OGCG2:         %[[VAL:.*]] = load i32, ptr %[[RET]]
// OGCG2:         ret i32 %[[VAL]]
