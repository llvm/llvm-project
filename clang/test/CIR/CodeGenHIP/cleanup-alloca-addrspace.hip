// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s --check-prefix=CIR-FLAT
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// Check that alloca created for cleanup destination slot for a function has
// correct address space after flatten cfg. This is important for address-space
// aware targets like amdgpu.

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z1fv
// CIR-FLAT:         cir.alloca "__cleanup_dest_slot" {{.*}} : !cir.ptr<!s32i, target_address_space(5)>
// CIR-FLAT:         cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_S, target_address_space(5)>
// CIR-FLAT:         cir.alloca "nrvo" {{.*}} : !cir.ptr<!cir.bool, target_address_space(5)>

// CIR-FLAT-LABEL: cir.func {{.*}} @_ZN1SD1Ev
// CIR-FLAT:         cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>

// CIR-FLAT-LABEL: cir.func {{.*}} @_ZN1SD2Ev
// CIR-FLAT:         cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>, target_address_space(5)>

// LLVM-LABEL: define {{.*}} @_Z1fv
// LLVM:         alloca i32, align 4, addrspace(5)
// LLVM:         alloca %struct.S, align 1, addrspace(5)
// LLVM:         alloca i8, align 1, addrspace(5)

// LLVM-LABEL: define {{.*}} @_ZN1SD1Ev
// LLVM:         alloca ptr, align 8, addrspace(5)

// LLVM-LABEL: define {{.*}} @_ZN1SD2Ev
// LLVM:         alloca ptr, align 8, addrspace(5)

struct S {
  __attribute__((device)) ~S() {}
};

__attribute__((device)) S f() {
  S s;
  return s;
}
