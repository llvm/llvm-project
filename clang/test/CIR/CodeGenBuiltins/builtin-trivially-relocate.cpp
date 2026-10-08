// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

typedef __SIZE_TYPE__ size_t;

struct S {
  int a;
  int b;
};

// CIR-LABEL: cir.func {{.*}} @_Z8relocateP1SS0_m
// CIR:         %[[DEST:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[SOURCE:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         %[[EIGHT:.*]] = cir.const #cir.int<8> : !u64i
// CIR:         %[[CONST_SIZE:.*]] = cir.mul %[[ONE]], %[[EIGHT]] : !u64i
// CIR:         %[[DEST_VOID:.*]] = cir.cast bitcast %[[DEST]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         %[[SOURCE_VOID:.*]] = cir.cast bitcast %[[SOURCE]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         cir.libc.memmove %[[CONST_SIZE]] bytes from %[[SOURCE_VOID]] to %[[DEST_VOID]] : !cir.ptr<!void>, !u64i
// CIR:         %[[DYNAMIC_DEST:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[DYNAMIC_SOURCE:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[COUNT:.*]] = cir.load {{.*}} : !cir.ptr<!u64i>, !u64i
// CIR:         %[[DYNAMIC_EIGHT:.*]] = cir.const #cir.int<8> : !u64i
// CIR:         %[[DYNAMIC_SIZE:.*]] = cir.mul %[[COUNT]], %[[DYNAMIC_EIGHT]] : !u64i
// CIR:         %[[DYNAMIC_DEST_VOID:.*]] = cir.cast bitcast %[[DYNAMIC_DEST]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         %[[DYNAMIC_SOURCE_VOID:.*]] = cir.cast bitcast %[[DYNAMIC_SOURCE]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:         cir.libc.memmove %[[DYNAMIC_SIZE]] bytes from %[[DYNAMIC_SOURCE_VOID]] to %[[DYNAMIC_DEST_VOID]] : !cir.ptr<!void>, !u64i
// CIR:         cir.store %[[DYNAMIC_DEST]], {{.*}} : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>

// LLVM-LABEL: define {{.*}} ptr @_Z8relocateP1SS0_m
// LLVM:         call void @llvm.memmove.p0.p0.i64({{.*}}i64 8, i1 false)
// LLVM:         %[[COUNT:.*]] = load i64, ptr {{.*}}
// LLVM:         %[[SIZE:.*]] = mul i64 %[[COUNT]], 8
// LLVM:         call void @llvm.memmove.p0.p0.i64({{.*}}i64 %[[SIZE]], i1 false)
// LLVM:         ret ptr
S *relocate(S *dest, S *source, size_t count) {
  __builtin_trivially_relocate(dest, source, 1);
  return __builtin_trivially_relocate(dest, source, count);
}

struct Empty {};

// CIR-LABEL: cir.func {{.*}} @_Z9reproduceP5Empty
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         %[[ELEMENT_SIZE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         %[[SIZE:.*]] = cir.mul %[[ONE]], %[[ELEMENT_SIZE]] : !u64i
// CIR:         cir.libc.memmove %[[SIZE]] bytes

// LLVM-LABEL: define {{.*}} void @_Z9reproduceP5Empty
// LLVM:         call void @llvm.memmove.p0.p0.i64({{.*}}i64 1, i1 false)
void reproduce(Empty *ptr) {
  __builtin_trivially_relocate(ptr, ptr, 1);
}
