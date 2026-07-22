// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef struct S {
  int x;
} S;

void non_atomic_to_atomic_cast() {
  S s;
  _Atomic(S) as =  s;
}

// CIR: %[[S_ADDR:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_S>
// CIR: %[[SA_ADDR:.*]] = cir.alloca "as" {{.*}} init : !cir.ptr<!rec_S>
// CIR: cir.copy %[[S_ADDR]] align(4) to %[[SA_ADDR]] align(4) : !cir.ptr<!rec_S>

// LLVM:  %[[S_ADDR:.*]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[SA_ADDR:.*]] = alloca %struct.S, i64 1, align 4
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[SA_ADDR]], ptr align 4 %[[S_ADDR]], i64 4, i1 false)

// OGCG: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[SA_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[SA_ADDR]], ptr align 4 %[[S_ADDR]], i64 4, i1 false)

void atomic_to_non_atomic_cast() {
  _Atomic S as;
  S s;
  s = as;
}

// CIR: %[[AS_ADDR:.*]] = cir.alloca "as" {{.*}} : !cir.ptr<!rec_S>
// CIR: %[[S_ADDR:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_S>
// CIR: %[[SA_PTR:.*]] = cir.cast bitcast %[[AS_ADDR]] : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
// CIR: %[[ATOMIC_LOAD:.*]] = cir.load {{.*}} atomic(seq_cst) %[[SA_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[S_PTR:.*]] = cir.cast bitcast %[[S_ADDR]] : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
// CIR: cir.store {{.*}} %[[ATOMIC_LOAD]], %[[S_PTR]] : !u32i, !cir.ptr<!u32i>

// LLVM: %[[AS_ADDR:.*]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[S_ADDR:.*]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[ATOMIC_LOAD:.*]] = load atomic i32, ptr %[[AS_ADDR]] seq_cst, align 4
// LLVM: store i32 %[[ATOMIC_LOAD]], ptr %[[S_ADDR]], align 4

// OGCG: %[[AS_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[ATOMIC_LOAD:.*]] = load atomic i32, ptr %[[AS_ADDR]] seq_cst, align 4
// OGCG: store i32 %[[ATOMIC_LOAD]], ptr %[[S_ADDR]], align 4
