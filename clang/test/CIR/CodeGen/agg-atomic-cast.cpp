// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int data[4];
};

void non_atomic_to_atomic_cast() {
  S s;
  _Atomic(S) as =  s;
}

// CIR: %[[S_ADDR:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_S>
// CIR: %[[SA_ADDR:.*]] = cir.alloca "as" {{.*}} init : !cir.ptr<!rec_S>
// CIR: cir.copy %[[S_ADDR]] to %[[SA_ADDR]] : !cir.ptr<!rec_S>

// LLVM:  %[[S_ADDR:.*]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[SA_ADDR:.*]] = alloca %struct.S, i64 1, align 16
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr %[[SA_ADDR]], ptr %[[S_ADDR]], i64 16, i1 false)

// OGCG: %[[S_ADDR:.*]] = alloca %struct.S, align 4
// OGCG: %[[SA_ADDR:.*]] = alloca %struct.S, align 16
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[SA_ADDR]], ptr align 4 %[[S_ADDR]], i64 16, i1 false)
