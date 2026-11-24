// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT,CIR-FLATO0
// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -x hip -fclangir -fcuda-is-device -O1 -emit-cir %s -o %to1.cir
// RUN: cir-opt --cir-flatten-cfg %to1.cir -o %t-flato1.cir
// RUN: FileCheck --input-file=%t-flato1.cir %s --check-prefix=CIR-FLAT,CIR-FLATO1
// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -x hip -fclangir -fcuda-is-device -O1 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVMOPT

// Check that alloca created for cleanup destination slot for a function has
// correct address space after flatten cfg. This is important for address-space
// aware targets like amdgpu.

// S is not trivially destructible, so f returns it through a generic sret
// pointer rather than a __retval alloca.
// CIR-FLAT-LABEL: cir.func {{.*}} @_Z1fv(%arg0: !cir.ptr<!rec_S> {{.*}}llvm.sret = !rec_S
// CIR-FLAT:       cir.alloca "__cleanup_dest_slot" {{.*}} : !cir.ptr<!s32i, target_address_space(5)>
// CIR-FLAT:       cir.alloca "nrvo" {{.*}} : !cir.ptr<!cir.bool, target_address_space(5)>

// CIR-FLATO0-LABEL: cir.func {{.*}} @_ZN1SD1Ev
// CIR-FLATO0:       cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>

// CIR-FLATO0-LABEL: cir.func {{.*}} @_ZN1SD2Ev
// CIR-FLATO0:       cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>

// CIR-FLATO1-LABEL: cir.func {{.*}} @_Z2f1i
// CIR-FLATO1:       cir.alloca "__ret_operand_tmp" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-FLATO1:       cir.alloca "__cleanup_dest_slot" align(4) cleanup_dest_slot : !cir.ptr<!s32i, target_address_space(5)>
// CIR-FLATO1:       cir.alloca "__retval" align(4) : !cir.ptr<!rec_S1, target_address_space(5)>
// CIR-FLATO1:       cir.alloca "a" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
// CIR-FLATO1:       cir.alloca "c" align(4) init const : !cir.ptr<!s32i, target_address_space(5)>

// LLVM-LABEL: define {{.*}} void @_Z1fv(ptr {{.*}}sret(%struct.S)
// LLVM:         alloca i32, align 4, addrspace(5)
// LLVM:         alloca i8, align 1, addrspace(5)

// LLVM-LABEL: define {{.*}} @_ZN1SD1Ev
// LLVM:         alloca ptr, align 8, addrspace(5)

// LLVM-LABEL: define {{.*}} @_ZN1SD2Ev
// LLVM:         alloca ptr, align 8, addrspace(5)

// Everything is optimized away but make sure verifiers agree with the lowering.
// LLVMOPT-LABEL: define {{.*}} i32 @_Z2f1i(i32 noundef returned %0)
// LLVMOPT-NEXT:     ret i32 %0

struct S {
  __attribute__((device)) ~S() {}
};

__attribute__((device)) S f() {
  S s;
  return s;
}

struct S1 { int x; };
__attribute__((device)) S1 f1(int a) {
  const int c = a;
  return {c};
}
