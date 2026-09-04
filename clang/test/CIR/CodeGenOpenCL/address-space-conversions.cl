// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir

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
