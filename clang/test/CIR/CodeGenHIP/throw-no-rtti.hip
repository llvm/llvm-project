// RUN: %clang_cc1 -std=c++17 -triple amdgcn-amd-amdhsa -x hip \
// RUN:   -fcuda-is-device --hipstdpar -fcxx-exceptions -fexceptions \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple amdgcn-amd-amdhsa -x hip \
// RUN:   -fcuda-is-device --hipstdpar -fcxx-exceptions -fexceptions \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple amdgcn-amd-amdhsa -x hip \
// RUN:   -fcuda-is-device --hipstdpar -fcxx-exceptions -fexceptions \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

struct X {
  ~X();
};

struct Guard {
  ~Guard();
};

inline void throw_int() {
  throw 1;
}

inline void throw_object() {
  throw X{};
}

inline void throw_with_cleanup() {
  Guard guard;
  throw 2;
}

inline void throw_object_with_cleanup() {
  Guard guard;
  throw X{};
}

int main() {
  throw_int();
  throw_object();
  throw_with_cleanup();
  throw_object_with_cleanup();
}

// Device-side HIP compilation disables RTTI. The Itanium throw ABI still
// requires a type-info argument, so use a null pointer in the target's global
// address space instead of trying to create a reference to a missing RTTI
// global.

// CIR: module
// CIR-NOT: cir.global {{.*}} @_ZTI

// CIR-LABEL: cir.func{{.*}} @_Z9throw_intv()
// CIR: %[[EXCEPTION:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR: %[[TYPE_INFO:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i, target_address_space(1)>
// CIR: %[[DTOR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR: cir.throw %[[EXCEPTION]], %[[TYPE_INFO]], %[[DTOR]] : !cir.ptr<!s32i>, !cir.ptr<!u8i, target_address_space(1)>, !cir.ptr<!void>

// CIR-LABEL: cir.func{{.*}} @_Z12throw_objectv()
// CIR: %[[EXCEPTION:.*]] = cir.alloc.exception 1 -> !cir.ptr<!rec_X>
// CIR: %[[TYPE_INFO:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i, target_address_space(1)>
// CIR: %[[DTOR_ADDR:.*]] = cir.get_global @_ZN1XD1Ev
// CIR: %[[DTOR:.*]] = cir.cast bitcast %[[DTOR_ADDR]] : {{.*}} -> !cir.ptr<!void>
// CIR: cir.throw %[[EXCEPTION]], %[[TYPE_INFO]], %[[DTOR]] : !cir.ptr<!rec_X>, !cir.ptr<!u8i, target_address_space(1)>, !cir.ptr<!void>

// LLVM-LABEL: define{{.*}} void @_Z9throw_intv()
// LLVM: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr null)
// LLVM-LABEL: define{{.*}} void @_Z12throw_objectv()
// LLVM: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr @_ZN1XD1Ev)
// LLVM-LABEL: define{{.*}} void @_Z18throw_with_cleanupv()
// LLVM: invoke void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr null)
// LLVM-LABEL: define{{.*}} void @_Z25throw_object_with_cleanupv()
// LLVM: invoke void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr @_ZN1XD1Ev)
// LLVM: declare void @__cxa_throw(ptr, ptr addrspace(1), ptr)

// OGCG-LABEL: define{{.*}} void @_Z9throw_intv()
// OGCG: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr null)
// OGCG-LABEL: define{{.*}} void @_Z12throw_objectv()
// OGCG: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr @_ZN1XD1Ev)
// OGCG-LABEL: define{{.*}} void @_Z18throw_with_cleanupv()
// OGCG: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr null)
// OGCG-LABEL: define{{.*}} void @_Z25throw_object_with_cleanupv()
// OGCG: call void @__cxa_throw(ptr %{{.*}}, ptr addrspace(1) null, ptr @_ZN1XD1Ev)
// OGCG: declare void @__cxa_throw(ptr, ptr addrspace(1), ptr)
