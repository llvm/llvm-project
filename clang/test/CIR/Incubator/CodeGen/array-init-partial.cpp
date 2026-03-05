// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

struct HasDtor {
  HasDtor();
  HasDtor(int x);
  ~HasDtor();
  int value;
};

// Test: Partial explicit initialization (3 of 5 elements)
// This should create endOfInit tracking for exception safety

// CIR-LABEL: @_Z{{.*}}test_partial
// CIR: arrayinit.endOfInit
// CIR: cir.call @_ZN7HasDtorC1Ei
// CIR: cir.call @_ZN7HasDtorC1Ei
// CIR: cir.call @_ZN7HasDtorC1Ei
// CIR: cir.do {
// CIR: cir.call @_ZN7HasDtorC1Ev
// CIR: cir.do {
// CIR: cir.call @_ZN7HasDtorD1Ev

// LLVM-LABEL: define {{.*}}void @_Z12test_partialv()
// LLVM: %[[ENDOFINIT:.*]] = alloca ptr
// LLVM: %[[ARR:.*]] = alloca [5 x %struct.HasDtor]
// LLVM: store ptr %{{.*}}, ptr %[[ENDOFINIT]]
// LLVM: call void @_ZN7HasDtorC1Ei(ptr %{{.*}}, i32 1)
// LLVM: store ptr %{{.*}}, ptr %[[ENDOFINIT]]
// LLVM: call void @_ZN7HasDtorC1Ei(ptr %{{.*}}, i32 2)
// LLVM: store ptr %{{.*}}, ptr %[[ENDOFINIT]]
// LLVM: call void @_ZN7HasDtorC1Ei(ptr %{{.*}}, i32 3)
// LLVM: call void @_ZN7HasDtorC1Ev(ptr %{{.*}})
// LLVM: call void @_ZN7HasDtorD1Ev(ptr %{{.*}})

// OGCG-LABEL: define {{.*}}void @_Z12test_partialv()
// OGCG: %[[ARR:.*]] = alloca [5 x %struct.HasDtor]
// OGCG: call void @_ZN7HasDtorC1Ei(ptr {{.*}} %[[ARR]], i32 {{.*}} 1)
// OGCG: %[[ELEM1:.*]] = getelementptr {{.*}} %struct.HasDtor, ptr %[[ARR]], i64 1
// OGCG: call void @_ZN7HasDtorC1Ei(ptr {{.*}} %[[ELEM1]], i32 {{.*}} 2)
// OGCG: %[[ELEM2:.*]] = getelementptr {{.*}} %struct.HasDtor, ptr %[[ARR]], i64 2
// OGCG: call void @_ZN7HasDtorC1Ei(ptr {{.*}} %[[ELEM2]], i32 {{.*}} 3)
// OGCG: call void @_ZN7HasDtorC1Ev(ptr {{.*}})
// OGCG: call void @_ZN7HasDtorD1Ev(ptr {{.*}})

void test_partial() {
  HasDtor arr[5] = {HasDtor(1), HasDtor(2), HasDtor(3)};
}
