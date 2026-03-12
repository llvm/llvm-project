// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

struct S { void do_thing(); };

void test_const_cast(const S &s) {
  const_cast<S&>(s).do_thing();
}
// CIR: cir.func {{.*}}@_Z15test_const_castRK1S(%[[ARG:.*]]: !cir.ptr<!rec_S>{{.*}}) {
// CIR-NEXT:   %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init, const]
// CIR-NEXT:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   %[[ARG_LOAD:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR-NEXT:   cir.call @_ZN1S8do_thingEv(%[[ARG_LOAD]]) 

// LLVM: define {{.*}}void @_Z15test_const_castRK1S(ptr{{.*}}%[[ARG:.*]])
// LLVM: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM-NEXT: store ptr %[[ARG]], ptr %[[ARG_ALLOCA]]
// LLVM-NEXT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM-NEXT: call void @_ZN1S8do_thingEv(ptr {{.*}}%[[ARG_LOAD]])

void call_with_ri_cast(S*&);
void call_with_ri_cast(int*&);
void test_reinterpet_cast(void *&data) {
  call_with_ri_cast(reinterpret_cast<S*&>(data));
  call_with_ri_cast(reinterpret_cast<int*&>(data));
}
// CIR: cir.func {{.*}}@_Z20test_reinterpet_castRPv(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!void>>{{.*}}) {
// CIR-NEXT:   %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, ["data", init, const]
// CIR-NEXT:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR-NEXT:   %[[ARG_LOAD:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT:   %[[RI_CAST:.*]] = cir.cast bitcast %[[ARG_LOAD]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   cir.call @_Z17call_with_ri_castRP1S(%[[RI_CAST]])
// CIR-NEXT:   %[[ARG_LOAD:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT:   %[[RI_CAST:.*]] = cir.cast bitcast %[[ARG_LOAD]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   cir.call @_Z17call_with_ri_castRPi(%[[RI_CAST]])

// LLVM: define dso_local void @_Z20test_reinterpet_castRPv(ptr {{.*}}%[[ARG:.*]])
// LLVM: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM-NEXT: store ptr %[[ARG]], ptr %[[ARG_ALLOCA]]
// LLVM-NEXT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM-NEXT: call void @_Z17call_with_ri_castRP1S(ptr {{.*}}%[[ARG_LOAD]])
// LLVM-NEXT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM-NEXT: call void @_Z17call_with_ri_castRPi(ptr {{.*}}%[[ARG_LOAD]])
