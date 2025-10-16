// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Should constant initialize global with constant address.
int var = 1;
int *constAddr = &var;

// CIR: cir.global external @constAddr = #cir.global_view<@var> : !cir.ptr<!s32i>

// LLVM: @constAddr = global ptr @var, align 8

// OGCG: @constAddr = global ptr @var, align 8

// Should constant initialize global with constant address.
int f();
int (*constFnAddr)() = f;

// CIR: cir.global external @constFnAddr = #cir.global_view<@_Z1fv> : !cir.ptr<!cir.func<() -> !s32i>>

// LLVM: @constFnAddr = global ptr @_Z1fv, align 8

// OGCG: @constFnAddr = global ptr @_Z1fv, align 8

int arr[4][16];
int *constArrAddr = &arr[2][1];

// CIR: cir.global external @constArrAddr = #cir.global_view<@arr, [2 : i32, 1 : i32]> : !cir.ptr<!s32i>

// The 'inbounds' and 'nuw' flags are inferred by LLVM's constant folder. The
// same flags show up at -O1 in OGCG.
// LLVM: @constArrAddr = global ptr getelementptr inbounds nuw (i8, ptr @arr, i64 132), align 8

// OGCG: @constArrAddr = global ptr getelementptr (i8, ptr @arr, i64 132), align 8
