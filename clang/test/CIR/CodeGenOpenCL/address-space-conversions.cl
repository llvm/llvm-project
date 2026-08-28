// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -fclangir -emit-llvm -O0 %s -o %t.cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.cir.ll
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -emit-llvm -O0 %s -o %t.ogcg.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ogcg.ll

void address_space_conversions(global int *global_ptr,
                               generic int *generic_ptr,
                               private int *private_ptr) {
  generic_ptr = global_ptr;
  generic_ptr = private_ptr;
  global_ptr = (global int *)generic_ptr;
}

// CIR-LABEL: cir.func dso_local @address_space_conversions
// CIR: cir.cast address_space
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>

// LLVM-LABEL: define {{.*}}void @address_space_conversions
// LLVM-SAME: ptr addrspace(1) noundef
// LLVM-SAME: ptr addrspace(4) noundef
// LLVM-SAME: ptr noundef
// LLVM: addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(4)
// LLVM: addrspacecast ptr %{{.*}} to ptr addrspace(4)
// LLVM: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(1)
