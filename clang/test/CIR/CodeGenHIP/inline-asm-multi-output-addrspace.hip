// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// The results of a multi-output asm are read back from a private (AS 5)
// temporary, so the member pointers must stay in AS 5.

__attribute__((device)) int two_outputs() {
  int a, b;
  asm("v_mov_b32 %0, 1\n v_mov_b32 %1, 2" : "=v"(a), "=v"(b));
  return a + b;
}

// CIR-LABEL: cir.func {{.*}}@_Z11two_outputsv
// CIR: %[[RES:.*]] = cir.alloca "__asm_result" {{.*}} : !cir.ptr<!rec_anon_struct, target_address_space(5)>
// CIR: cir.get_member %[[RES]][0] {name = ""} : !cir.ptr<!rec_anon_struct, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR: cir.get_member %[[RES]][1] {name = ""} : !cir.ptr<!rec_anon_struct, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>

// LLVM-LABEL: define {{.*}}@_Z11two_outputsv
// LLVM: getelementptr inbounds nuw { i32, i32 }, ptr addrspace(5) %{{.*}}, i32 0, i32 0
// LLVM: getelementptr inbounds nuw { i32, i32 }, ptr addrspace(5) %{{.*}}, i32 0, i32 1
