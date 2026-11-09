// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// The device global is in the global address space and its destructor takes
// `this` in the generic one. All __cxa_atexit arguments are generic. SYCL
// only allows device code to use const, constant-initialized globals.

struct S {
  ~S();
  int x;
};

const S s{1};

[[clang::sycl_external]] int get() { return s.x; }

// CIR: cir.global "private" hidden external target_address_space(1) @__dso_handle : !u8i
// CIR: cir.func private @__cxa_atexit(!cir.ptr<!cir.func<(!cir.ptr<!void>)>, target_address_space(4)>, !cir.ptr<!void, target_address_space(4)>, !cir.ptr<!u8i, target_address_space(4)>) -> !s32i
// CIR: cir.global "private" internal dso_local target_address_space(1) @_ZL1s = #cir.const_record<{#cir.int<1> : !s32i}> : !rec_S

// CIR: cir.func internal private @__cxx_global_var_init()
// CIR:   %[[S:.*]] = cir.get_global @_ZL1s : !cir.ptr<!rec_S, target_address_space(1)>
// CIR:   %[[S_GEN:.*]] = cir.cast address_space %[[S]] : !cir.ptr<!rec_S, target_address_space(1)> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR:   %[[DTOR:.*]] = cir.get_global @_ZN1SD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_S, target_address_space(4)>)>>
// CIR:   %[[DTOR_VOID:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_S, target_address_space(4)>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[DTOR_GEN:.*]] = cir.cast address_space %[[DTOR_VOID]] : !cir.ptr<!cir.func<(!cir.ptr<!void>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>, target_address_space(4)>
// CIR:   %[[S_VOID:.*]] = cir.cast bitcast %[[S_GEN]] : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!void, target_address_space(4)>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<!u8i, target_address_space(1)>
// CIR:   %[[HANDLE_GEN:.*]] = cir.cast address_space %[[HANDLE]] : !cir.ptr<!u8i, target_address_space(1)> -> !cir.ptr<!u8i, target_address_space(4)>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_GEN]], %[[S_VOID]], %[[HANDLE_GEN]])

// LLVM-DAG: @_ZL1s = internal addrspace(1) global %struct.S { i32 1 }, align 4
// LLVM-DAG: @__dso_handle = external hidden addrspace(1) global i8

// LLVM: define internal {{.*}}void @__cxx_global_var_init()
// LLVM:   call {{.*}}i32 @__cxa_atexit(ptr addrspace(4) addrspacecast (ptr @_ZN1SD1Ev to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL1s to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__dso_handle to ptr addrspace(4)))
