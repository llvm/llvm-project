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

// LLVM: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// LLVM: %[[SA_ADDR:.*]] = alloca %struct.S, align 4
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

// LLVM: %[[AS_ADDR:.*]] = alloca %struct.S, align 4
// LLVM: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// LLVM: %[[ATOMIC_LOAD:.*]] = load atomic i32, ptr %[[AS_ADDR]] seq_cst, align 4
// LLVM: store i32 %[[ATOMIC_LOAD]], ptr %[[S_ADDR]], align 4

// OGCG: %[[AS_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[ATOMIC_LOAD:.*]] = load atomic i32, ptr %[[AS_ADDR]] seq_cst, align 4
// OGCG: store i32 %[[ATOMIC_LOAD]], ptr %[[S_ADDR]], align 4

struct T {
  char a, b, c;
}; // size 3 => padded atomic representation


struct T load_atomic_struct() { 
  _Atomic(struct T) a;
  return a; 
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_T>
// CIR: %[[RET_VAL_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_T>
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_anon_struct>
// CIR: %[[NON_ATOMIC_TMP_ADDR:.*]] = cir.alloca "tmp" {{.*}} : !cir.ptr<!rec_anon_struct>
// CIR: %[[A_U32:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR: %[[TMP_ATOMIC_A:.*]] = cir.load {{.*}} atomic(seq_cst) %[[A_U32]] : !cir.ptr<!u32i>, !u32i
// CIR: %[[NON_ATOMIC_TMP:.*]] = cir.cast bitcast %[[NON_ATOMIC_TMP_ADDR]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u32i>
// CIR: cir.store {{.*}} %[[TMP_ATOMIC_A]], %[[NON_ATOMIC_TMP]] : !u32i, !cir.ptr<!u32i>
// CIR: %[[VALUE_ADDR:.*]] = cir.get_member %[[NON_ATOMIC_TMP_ADDR]][0] {name = "value_addr"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CIR: cir.copy %[[VALUE_ADDR]] {{.*}} to %[[RET_VAL_ADDR]] {{.*}} : !cir.ptr<!rec_T>
// CIR: %[[TMP_RET_VAL:.*]] = cir.load %[[RET_VAL_ADDR]] : !cir.ptr<!rec_T>, !rec_T
// CIR: cir.store %[[TMP_RET_VAL]], %[[RET_ADDR]] : !rec_T, !cir.ptr<!rec_T>
// CIR: %[[RET_ADDR_U64:.*]] = cir.cast bitcast %[[RET_ADDR]] : !cir.ptr<!rec_T> -> !cir.ptr<!cir.int<u, 24>>
// CIR: %[[TMP_RET:.*]] = cir.load %[[RET_ADDR_U64]] : !cir.ptr<!cir.int<u, 24>>, !cir.int<u, 24>
// CIR: cir.return %[[TMP_RET]] : !cir.int<u, 24>

// The difference between LLVM and OGCG in type from struct.T to i24 is due to missing ABI lowering.

// LLVM: %[[RET_ADDR:.*]] = alloca %struct.T, align 4
// LLVM: %[[RET_VAL_ADDR:.*]] = alloca %struct.T, align 1
// LLVM: %[[A_ADDR:.*]] = alloca { %struct.T, [1 x i8] }, align 4
// LLVM: %[[NON_ATOMIC_TMP:.*]] = alloca { %struct.T, [1 x i8] }, align 4
// LLVM: %[[TMP_A:.*]] = load atomic i32, ptr %[[A_ADDR]] seq_cst, align 4
// LLVM: store i32 %[[TMP_A]], ptr %[[NON_ATOMIC_TMP]], align 4
// LLVM: %[[NON_ATOMIC_PTR:.*]] = getelementptr inbounds nuw { %struct.T, [1 x i8] }, ptr %[[NON_ATOMIC_TMP]], i32 0, i32 0
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[RET_VAL_ADDR]], ptr align 4 %[[NON_ATOMIC_PTR]], i64 3, i1 false)
// LLVM: %[[TMP_RET_VAL:.*]] = load %struct.T, ptr %[[RET_VAL_ADDR]], align 1
// LLVM: store %struct.T %[[TMP_RET_VAL]], ptr %[[RET_ADDR]], align 1
// LLVM: %[[TMP_RET:.*]] = load i24, ptr %[[RET_ADDR]], align 4
// LLVM: ret i24 %[[TMP_RET]]

// OGCG: %[[RET_VAL_ADDR:.*]] = alloca %struct.T, align 1
// OGCG: %[[A_ADDR:.*]] = alloca { %struct.T, [1 x i8] }, align 4
// OGCG: %[[NON_ATOMIC_TMP:.*]] = alloca { %struct.T, [1 x i8] }, align 4
// OGCG: %[[RET_ADDR:.*]] = alloca i24, align 4
// OGCG: %[[TMP_A:.*]] = load atomic i32, ptr %[[A_ADDR]] seq_cst, align 4
// OGCG: store i32 %[[TMP_A]], ptr %[[NON_ATOMIC_TMP]], align 4
// OGCG: %[[NON_ATOMIC_PTR:.*]] = getelementptr inbounds nuw { %struct.T, [1 x i8] }, ptr %atomic-to-nonatomic.temp, i32 0, i32 0
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[RET_VAL_ADDR]], ptr align 4 %[[NON_ATOMIC_PTR]], i64 3, i1 false)
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[RET_ADDR]], ptr align 1 %[[RET_VAL_ADDR]], i64 3, i1 false)
// OGCG: %[[TMP_RET:.*]] = load i24, ptr %[[RET_ADDR]], align 4
// OGCG: ret i24 %[[TMP_RET]]
