// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir -check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll -check-prefix=LLVM %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=OGCG %s

typedef __typeof__(sizeof(0)) size_t;

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw();

struct A { A(); ~A(); };

void test_reserved_placement_new(void *p) {
  new (p) A();
}

// CIR-LABEL:   cir.func {{.*}} @_Z27test_reserved_placement_newPv(
// CIR-SAME:                                   %[[ARG0:.*]]: !cir.ptr<!void>
// CIR:           %[[P:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
// CIR:           cir.store %[[ARG0]], %[[P]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:           %[[PTR:.*]] = cir.load{{.*}} %[[P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:           %[[PTR_A:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!rec_A>
// CIR:           cir.call @_ZN1AC1Ev(%[[PTR_A]]) : (!cir.ptr<!rec_A>) -> ()

// LLVM-LABEL: define dso_local void @_Z27test_reserved_placement_newPv(
// LLVM-SAME:                                   ptr %[[ARG0:.*]]
// LLVM:         %[[P:.*]] = alloca ptr
// LLVM:         store ptr %[[ARG0:.*]], ptr %[[P]]
// LLVM:         %[[PTR:.*]] = load ptr, ptr %[[P]]
// LLVM:         call void @_ZN1AC1Ev(ptr %[[PTR]])

// OGCG-LABEL: define dso_local void @_Z27test_reserved_placement_newPv(
// OGCG-SAME:                                   ptr {{.*}} %[[ARG0:.*]]
// OGCG:         %[[P:.*]] = alloca ptr
// OGCG:         store ptr %[[ARG0:.*]], ptr %[[P]]
// OGCG:         %[[PTR:.*]] = load ptr, ptr %[[P]]
// OGCG:         call void @_ZN1AC1Ev(ptr {{.*}} %[[PTR]])
