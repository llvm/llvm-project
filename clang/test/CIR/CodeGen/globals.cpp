// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

// Should constant initialize global with constant address.
int var = 1;
int *constAddr = &var;

// CIR: cir.global external @constAddr = #cir.global_view<@var> : !cir.ptr<!s32i>

// LLVM: @constAddr = global ptr @var, align 8

// Should constant initialize global with constant address.
int f();
int (*constFnAddr)() = f;

// CIR: cir.global external @constFnAddr = #cir.global_view<@_Z1fv> : !cir.ptr<!cir.func<() -> !s32i>>

// LLVM: @constFnAddr = global ptr @_Z1fv, align 8
