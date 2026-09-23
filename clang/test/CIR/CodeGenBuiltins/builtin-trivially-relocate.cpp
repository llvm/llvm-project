// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM

typedef __SIZE_TYPE__ size_t;

struct S {
  int a;
  int b;
};

// The count is in elements, so it is scaled by sizeof(S) before the memmove.
S *test(S *source, S *dest, size_t count) {
  return __builtin_trivially_relocate(dest, source, count);
}

// CIR-LABEL: cir.func {{.*}} @_Z4testP1SS0_m(
// CIR:         %[[SOURCE_ADDR:.*]] = cir.alloca "source"
// CIR:         %[[DEST_ADDR:.*]] = cir.alloca "dest"
// CIR:         %[[DEST:.*]] = cir.load {{.*}} %[[DEST_ADDR]]
// CIR:         %[[SOURCE:.*]] = cir.load {{.*}} %[[SOURCE_ADDR]]
// CIR:         %[[COUNT:.*]] = cir.load {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR:         %[[ELT_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:         %[[SIZE:.*]] = cir.mul %[[COUNT]], %[[ELT_SIZE]] : !u64i
// CIR:         %[[SOURCE_VOID:.*]] = cir.cast bitcast %[[SOURCE]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         %[[DEST_VOID:.*]] = cir.cast bitcast %[[DEST]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         cir.libc.memmove %[[SIZE]] bytes from %[[SOURCE_VOID]] align(4) to %[[DEST_VOID]] align(4)
// CIR:         cir.store %[[DEST]], %{{.*}} : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>

// LLVM-LABEL: define {{.*}} ptr @_Z4testP1SS0_m(
// LLVM:         [[DEST:%.*]] = load ptr, ptr
// LLVM:         [[SOURCE:%.*]] = load ptr, ptr
// LLVM:         [[COUNT:%.*]] = load i64, ptr
// LLVM:         [[SIZE:%.*]] = mul i64 [[COUNT]], 8
// LLVM:         call void @llvm.memmove.p0.p0.i64(ptr align 4 [[DEST]], ptr align 4 [[SOURCE]], i64 [[SIZE]], i1 false)
