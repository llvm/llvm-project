// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

class x {
public: int operator=(int);
};
void a() {
  x a;
  a = 1u;
}

// CIR: cir.func private @_ZN1xaSEi(!cir.ptr<!rec_x>, !s32i)
// CIR: cir.func{{.*}} @_Z1av()
// CIR:   %[[A_ADDR:.*]] = cir.alloca !rec_x, !cir.ptr<!rec_x>, ["a"]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CIR:   %[[ONE_CAST:.*]] = cir.cast integral %[[ONE]] : !u32i -> !s32i
// CIR:   %[[RET:.*]] = cir.call @_ZN1xaSEi(%[[A_ADDR]], %[[ONE_CAST]]) : (!cir.ptr<!rec_x>, !s32i) -> !s32i

// LLVM: define{{.*}} @_Z1av(){{.*}}
// OGCG: define{{.*}} @_Z1av()

void f(int i, int j) {
  (i += j) = 17;
}

// CIR: cir.func{{.*}} @_Z1fii(%arg0: !s32i {{.*}}, %arg1: !s32i {{.*}})
// CIR:   %[[I_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   %[[J_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["j", init]
// CIR:   cir.store %arg0, %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store %arg1, %[[J_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[SEVENTEEN:.*]] = cir.const #cir.int<17> : !s32i
// CIR:   %[[J_LOAD:.*]] = cir.load align(4) %[[J_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[I_LOAD:.*]] = cir.load align(4) %[[I_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[ADD:.*]] = cir.binop(add, %[[I_LOAD]], %[[J_LOAD]]) nsw : !s32i
// CIR:   cir.store align(4) %[[ADD]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.store align(4) %[[SEVENTEEN]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// Ensure that we use memcpy when we would have selected a trivial assignment
// operator, even for a non-trivially-copyable type.
struct A {
  A &operator=(const A&);
};
struct B {
  B(const B&);
  B &operator=(const B&) = default;
  int n;
};
struct C {
  A a;
  B b[16];
};
void copy_c(C &c1, C &c2) {
  c1 = c2;
}

// CIR: cir.func private @_ZN1AaSERKS_(!cir.ptr<!rec_A>, !cir.ptr<!rec_A>) -> !cir.ptr<!rec_A>
// CIR: cir.func private @memcpy(!cir.ptr<!void>, !cir.ptr<!void>, !u64i) -> !cir.ptr<!void>

// Implicit assignment operator for C.

// CIR: cir.func comdat linkonce_odr @_ZN1CaSERKS_(%arg0: !cir.ptr<!rec_C> {{.*}}, %arg1: !cir.ptr<!rec_C> {{.*}}) -> !cir.ptr<!rec_C>
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR:   %[[ARG1_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["", init, const]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["__retval"]
// CIR:   cir.store %arg0, %[[THIS_ADDR]]
// CIR:   cir.store %arg1, %[[ARG1_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CIR:   %[[A_MEMBER:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CIR:   %[[ARG1_LOAD:.*]] = cir.load{{.*}} %[[ARG1_ADDR]]
// CIR:   %[[A_MEMBER_2:.*]] = cir.get_member %[[ARG1_LOAD]][0] {name = "a"}
// CIR:   %[[C_A:.*]] = cir.call @_ZN1AaSERKS_(%[[A_MEMBER]], %[[A_MEMBER_2]])
// CIR:   %[[B_MEMBER:.*]] = cir.get_member %[[THIS]][1] {name = "b"}
// CIR:   %[[B_VOID_PTR:.*]] = cir.cast bitcast %[[B_MEMBER]] : !cir.ptr<!cir.array<!rec_B x 16>> -> !cir.ptr<!void>
// CIR:   %[[RET_LOAD:.*]] = cir.load %[[ARG1_ADDR]]
// CIR:   %[[B_MEMBER_2:.*]] = cir.get_member %[[RET_LOAD]][1] {name = "b"}
// CIR:   %[[B_VOID_PTR_2:.*]] = cir.cast bitcast %[[B_MEMBER_2]] : !cir.ptr<!cir.array<!rec_B x 16>> -> !cir.ptr<!void>
// CIR:   %[[SIZE:.*]] = cir.const #cir.int<64> : !u64i
// CIR:   %[[COUNT:.*]] = cir.call @memcpy(%[[B_VOID_PTR]], %[[B_VOID_PTR_2]], %[[SIZE]])
// CIR:   cir.store %[[THIS]], %[[RET_ADDR]]
// CIR:   %[[RET_VAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RET_VAL]]

// CIR: cir.func{{.*}} @_Z6copy_cR1CS0_(%arg0: !cir.ptr<!rec_C> {{.*}}, %arg1: !cir.ptr<!rec_C> {{.*}})
// CIR:   %[[C1_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["c1", init, const]
// CIR:   %[[C2_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["c2", init, const]
// CIR:   cir.store %arg0, %[[C1_ADDR]]
// CIR:   cir.store %arg1, %[[C2_ADDR]]
// CIR:   %[[C2_LOAD:.*]] = cir.load{{.*}} %[[C2_ADDR]]
// CIR:   %[[C1_LOAD:.*]] = cir.load{{.*}} %[[C1_ADDR]]
// CIR:   %[[RET:.*]] = cir.call @_ZN1CaSERKS_(%[[C1_LOAD]], %[[C2_LOAD]])

struct D {
  D &operator=(const D&);
};
struct E {
  D &get_d_ref() { return d; }
private:
  D d;
};

void copy_ref_to_ref(E &e1, E &e2) {
  e1.get_d_ref() = e2.get_d_ref();
}

// The call to e2.get_d_ref() must occur before the call to e1.get_d_ref().

// CIR: cir.func{{.*}} @_Z15copy_ref_to_refR1ES0_(%arg0: !cir.ptr<!rec_E> {{.*}}, %arg1: !cir.ptr<!rec_E> {{.*}})
// CIR:   %[[E1_ADDR:.*]] = cir.alloca !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>, ["e1", init, const]
// CIR:   %[[E2_ADDR:.*]] = cir.alloca !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>, ["e2", init, const]
// CIR:   cir.store %arg0, %[[E1_ADDR]] : !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>
// CIR:   cir.store %arg1, %[[E2_ADDR]] : !cir.ptr<!rec_E>, !cir.ptr<!cir.ptr<!rec_E>>
// CIR:   %[[E2:.*]] = cir.load %[[E2_ADDR]]
// CIR:   %[[D2_REF:.*]] = cir.call @_ZN1E9get_d_refEv(%[[E2]])
// CIR:   %[[E1:.*]] = cir.load %[[E1_ADDR]]
// CIR:   %[[D1_REF:.*]] = cir.call @_ZN1E9get_d_refEv(%[[E1]])
// CIR:   %[[D1_REF_2:.*]] = cir.call @_ZN1DaSERKS_(%[[D1_REF]], %[[D2_REF]])
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z15copy_ref_to_refR1ES0_(ptr %[[ARG0:.*]], ptr %[[ARG1:.*]]){{.*}} {
// LLVM:   %[[E1_ADDR:.*]] = alloca ptr
// LLVM:   %[[E2_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[E1_ADDR]]
// LLVM:   store ptr %[[ARG1]], ptr %[[E2_ADDR]]
// LLVM:   %[[E2:.*]] = load ptr, ptr %[[E2_ADDR]]
// LLVM:   %[[D2_REF:.*]] = call ptr @_ZN1E9get_d_refEv(ptr %[[E2]])
// LLVM:   %[[E1:.*]] = load ptr, ptr %[[E1_ADDR]]
// LLVM:   %[[D1_REF:.*]] = call ptr @_ZN1E9get_d_refEv(ptr %[[E1]])
// LLVM:   %[[D1_REF_2:.*]] = call ptr @_ZN1DaSERKS_(ptr %[[D1_REF]], ptr %[[D2_REF]])

// OGCG: define{{.*}} void @_Z15copy_ref_to_refR1ES0_(ptr{{.*}} %[[ARG0:.*]], ptr{{.*}} %[[ARG1:.*]])
// OGCG:   %[[E1_ADDR:.*]] = alloca ptr
// OGCG:   %[[E2_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[E1_ADDR]]
// OGCG:   store ptr %[[ARG1]], ptr %[[E2_ADDR]]
// OGCG:   %[[E2:.*]] = load ptr, ptr %[[E2_ADDR]]
// OGCG:   %[[D2_REF:.*]] = call{{.*}} ptr @_ZN1E9get_d_refEv(ptr{{.*}} %[[E2]])
// OGCG:   %[[E1:.*]] = load ptr, ptr %[[E1_ADDR]]
// OGCG:   %[[D1_REF:.*]] = call{{.*}} ptr @_ZN1E9get_d_refEv(ptr{{.*}} %[[E1]])
// OGCG:   %[[D1_REF_2:.*]] = call{{.*}} ptr @_ZN1DaSERKS_(ptr{{.*}} %[[D1_REF]], ptr{{.*}} %[[D2_REF]])
