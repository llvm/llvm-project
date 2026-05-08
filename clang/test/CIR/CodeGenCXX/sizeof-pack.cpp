// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

template<class ...Ts> auto foo() {
  int values[sizeof...(Ts) + 1] = {Ts::value... };
  return sizeof...(Ts);
}
struct S1 {
  static constexpr int value = 1;
};

struct S2 {
  static constexpr int value = 2;
};

struct S3 {
  static constexpr int value = 3;
};

void test() {
  foo<>();
  foo<S1, S2, S3>();
}
// CIR-DAG: cir.global "private" constant cir_private @__const._Z3fooIJ2S12S22S3EEDav.values = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 4>
// LLVM-DAG: @__const._Z3fooIJ2S12S22S3EEDav.values = private constant [4 x i32] [i32 1, i32 2, i32 3, i32 0]
// OGCG-DAG: @__const._Z3fooIJ2S12S22S3EEDav.values = private {{.*}}constant [4 x i32] [i32 1, i32 2, i32 3, i32 0]
// CIR-DAG: cir.global "private" constant cir_private @__const._Z3fooIJEEDav.values = #cir.zero : !cir.array<!s32i x 1>
// LLVM-DAG: @__const._Z3fooIJEEDav.values = private constant [1 x i32] zeroinitializer

// CIR: cir.func {{.*}}@_Z3fooIJEEDav()
// CIR: %[[RETVAL:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["__retval"]
// CIR: %[[VAL_ARR:.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>, ["values", init]
// CIR: %[[GET_GLOB_VAL:.*]] = cir.get_global @__const._Z3fooIJEEDav.values : !cir.ptr<!cir.array<!s32i x 1>>
// CIR: cir.copy %[[GET_GLOB_VAL]] to %[[VAL_ARR]] : !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CIR: cir.store %[[ZERO]], %[[RETVAL]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[RET_LOAD:.*]] = cir.load %[[RETVAL]]  : !cir.ptr<!u64i>, !u64i
// CIR: cir.return %[[RET_LOAD]] : !u64i

// LLVM: define {{.*}}i64 @_Z3fooIJEEDav()
// LLVM: %[[RETVAL:.*]] = alloca i64
// LLVM: %[[VAL_ARR:.*]] = alloca [1 x i32]
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr %[[VAL_ARR]], ptr{{.*}} @__const._Z3fooIJEEDav.values, i64 4, i1 false)
// LLVM: store i64 0, ptr %[[RETVAL]]
// LLVM: %[[RET_LOAD:.*]] = load i64, ptr %[[RETVAL]]
// LLVM: ret i64 %[[RET_LOAD]]

// OGCG: define {{.*}}i64 @_Z3fooIJEEDav()
// OGCG: %[[VALUES:.*]] = alloca [1 x i32]
// OGCG: call void @llvm.memset.p0.i64(ptr {{.*}}%[[VALUES]], i8 0, i64 4, i1 false)
// OGCG: ret i64 0

// CIR: cir.func {{.*}}@_Z3fooIJ2S12S22S3EEDav()
// CIR: %[[RETVAL:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["__retval"]
// CIR: %[[VAL_ARR:.*]] = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["values", init]
// CIR: %[[GET_GLOB_VAL:.*]] = cir.get_global @__const._Z3fooIJ2S12S22S3EEDav.values : !cir.ptr<!cir.array<!s32i x 4>>
// CIR: cir.copy %[[GET_GLOB_VAL]] to %[[VAL_ARR]] : !cir.ptr<!cir.array<!s32i x 4>>
// CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !u64i
// CIR: cir.store %[[THREE]], %[[RETVAL]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[RET_LOAD:.*]] = cir.load %[[RETVAL]]  : !cir.ptr<!u64i>, !u64i
// CIR: cir.return %[[RET_LOAD]] : !u64i

// LLVM: define {{.*}}i64 @_Z3fooIJ2S12S22S3EEDav()
// LLVM: %[[RETVAL:.*]] = alloca i64
// LLVM: %[[VAL_ARR:.*]] = alloca [4 x i32]
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr %[[VAL_ARR]], ptr{{.*}} @__const._Z3fooIJ2S12S22S3EEDav.values, i64 16, i1 false)
// LLVM: store i64 3, ptr %[[RETVAL]]
// LLVM: %[[RET_LOAD:.*]] = load i64, ptr %[[RETVAL]]
// LLVM: ret i64 %[[RET_LOAD]]

// OGCG: define {{.*}}i64 @_Z3fooIJ2S12S22S3EEDav()
// OGCG: %[[VALUES:.*]] = alloca [4 x i32]
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VALUES]], ptr {{.*}}@__const._Z3fooIJ2S12S22S3EEDav.values, i64 16, i1 false)
// OGCG: ret i64 3
