// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

constexpr extern int cx_var = __builtin_is_constant_evaluated();

// CIR: cir.global {{.*}} @cx_var = #cir.int<1> : !s32i
// LLVM: @cx_var = {{.*}} i32 1
// OGCG: @cx_var = {{.*}} i32 1

int is_constant_evaluated() {
    return __builtin_is_constant_evaluated();
}

// CIR: cir.func @_Z21is_constant_evaluatedv() -> !s32i
// CIR: %[[ZERO:.+]] = cir.const #cir.int<0>

// LLVM: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// LLVM: %[[MEM:.+]] = alloca i32
// LLVM: store i32 0, ptr %[[MEM]]
// LLVM: %[[RETVAL:.+]] = load i32, ptr %[[MEM]]
// LLVM: ret i32 %[[RETVAL]]
// LLVM: }

// OGCG: define {{.*}}i32 @_Z21is_constant_evaluatedv()
// OGCG: ret i32 0
// OGCG: }
