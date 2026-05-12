// RUN: split-file %s %t

// 'main' implicitly returns 0 if it falls off the end of the function. CIR
// must initialize the return slot to 0 in the prologue so that loading it for
// the implicit return does not yield an undefined value.


//--- empty_body.c

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t/empty_body.c -o %t/empty_body.cir
// RUN: FileCheck --input-file=%t/empty_body.cir %t/empty_body.c --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/empty_body.c -o %t/empty_body-cir.ll
// RUN: FileCheck --input-file=%t/empty_body-cir.ll %t/empty_body.c --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/empty_body.c -o %t/empty_body.ll
// RUN: FileCheck --input-file=%t/empty_body.ll %t/empty_body.c --check-prefix=OGCG

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
// RUN: FileCheck --input-file=%t/nonempty_body.cir %t/nonempty_body.c --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %t/nonempty_body.c -o %t/nonempty_body-cir.ll
// RUN: FileCheck --input-file=%t/nonempty_body-cir.ll %t/nonempty_body.c --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t/nonempty_body.c -o %t/nonempty_body.ll
// RUN: FileCheck --input-file=%t/nonempty_body.ll %t/nonempty_body.c --check-prefix=OGCG

int g;

int main(void) {
  if (g != 0)
    return g;
}

// CIR-LABEL: cir.func{{.*}} @main
// CIR:         %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store %[[ZERO]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.if
// CIR:           %[[G_ADDR:.*]] = cir.get_global @g
// CIR:           %[[G:.*]] = cir.load{{.*}} %[[G_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:           cir.store %[[G]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR:           %[[EXPL:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR:           cir.return %[[EXPL]] : !s32i
// CIR:         %[[IMPL:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.return %[[IMPL]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @main(
// LLVM:         %[[RET:.*]] = alloca i32
// LLVM:         store i32 0, ptr %[[RET]]
// LLVM:         icmp ne
// LLVM:         %[[G:.*]] = load i32, ptr @g
// LLVM:         store i32 %[[G]], ptr %[[RET]]
// LLVM:         %[[EXPL:.*]] = load i32, ptr %[[RET]]
// LLVM:         ret i32 %[[EXPL]]
// LLVM:         %[[IMPL:.*]] = load i32, ptr %[[RET]]
// LLVM:         ret i32 %[[IMPL]]

// OGCG-LABEL: define{{.*}} i32 @main(
// OGCG:         %[[RET:.*]] = alloca i32
// OGCG:         store i32 0, ptr %[[RET]]
// OGCG:         icmp ne
// OGCG:         %[[G:.*]] = load i32, ptr @g
// OGCG:         store i32 %[[G]], ptr %[[RET]]
// OGCG:         %[[VAL:.*]] = load i32, ptr %[[RET]]
// OGCG:         ret i32 %[[VAL]]
