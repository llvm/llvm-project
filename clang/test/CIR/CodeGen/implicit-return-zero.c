// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// 'main' implicitly returns 0 if it falls off the end of the function. CIR
// must initialize the return slot to 0 in the prologue so that loading it for
// the implicit return does not yield an undefined value.

int main(void) {
}

// CIR-LABEL: cir.func{{.*}} @main
// CIR:         %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:    cir.store %[[ZERO]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR:         %[[LOAD:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    cir.return %[[LOAD]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @main(
// LLVM:         %[[RET:.*]] = alloca i32
// LLVM:         store i32 0, ptr %[[RET]]
// LLVM:         %[[LOAD:.*]] = load i32, ptr %[[RET]]
// LLVM:         ret i32 %[[LOAD]]

// OGCG-LABEL: define{{.*}} i32 @main(
// OGCG:         ret i32 0
