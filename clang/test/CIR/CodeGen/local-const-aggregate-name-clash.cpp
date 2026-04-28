// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Two distinct local aggregates in the same function may share a name (for
// example, arrays declared in different blocks). The per-function constant
// globals (@__const.<func>.<var>) materialized to back memcpy initialization
// must be disambiguated with a version suffix so that each cir.get_global
// agrees with the type and value of the global it references.

void use_array(const int *p, int n);

void f(bool which) {
  if (which) {
    int arr[] = {10, 20, 30, 40};
    use_array(arr, 4);
  } else {
    int arr[] = {50, 60};
    use_array(arr, 2);
  }
}

// CIR-DAG: cir.global "private" constant cir_private @[[GV0:.*]] = #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i, #cir.int<40> : !s32i]> : !cir.array<!s32i x 4>
// CIR-DAG: cir.global "private" constant cir_private @[[GV1:.*]] = #cir.const_array<[#cir.int<50> : !s32i, #cir.int<60> : !s32i]> : !cir.array<!s32i x 2>

// CIR: cir.func{{.*}} @_Z1fb
// CIR:   cir.get_global @[[GV0]] : !cir.ptr<!cir.array<!s32i x 4>>
// CIR:   cir.get_global @[[GV1]] : !cir.ptr<!cir.array<!s32i x 2>>

// LLVM-DAG: @[[GV0:.*]] = private constant [4 x i32] [i32 10, i32 20, i32 30, i32 40]
// LLVM-DAG: @[[GV1:.*]] = private constant [2 x i32] [i32 50, i32 60]

// LLVM: define{{.*}} @_Z1fb
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{[^,]+}}, ptr @[[GV0]], i64 16, i1 false)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{[^,]+}}, ptr @[[GV1]], i64 8, i1 false)

// OGCG-DAG: @[[GV0:.*]] = private unnamed_addr constant [4 x i32] [i32 10, i32 20, i32 30, i32 40]
// OGCG-DAG: @[[GV1:.*]] = private unnamed_addr constant [2 x i32] [i32 50, i32 60]

// OGCG: define{{.*}} @_Z1fb
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr {{[^,]+}}, ptr {{[^,]+}}@[[GV0]], i64 16, i1 false)
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr {{[^,]+}}, ptr {{[^,]+}}@[[GV1]], i64 8, i1 false)
