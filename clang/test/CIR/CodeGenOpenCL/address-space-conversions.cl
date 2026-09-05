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

// Alloca slots for each parameter (outer type is pointer-to-pointer, no outer
// address space yet — the inner pointer carries the OpenCL address space).
// CIR:   %[[GLOBAL_PTR_ALLOCA:.*]] = cir.alloca "global_ptr" {{.*}} : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>>
// CIR:   %[[GENERIC_PTR_ALLOCA:.*]] = cir.alloca "generic_ptr" {{.*}} : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>>
// CIR:   %[[PRIVATE_PTR_ALLOCA:.*]] = cir.alloca "private_ptr" {{.*}} : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_private)>>

// Each alloca is immediately cast to add offload_private on the outer pointer.
// All subsequent loads/stores go through these cast results.
// CIR:   %[[GLOBAL_SLOT:.*]] = cir.cast address_space %[[GLOBAL_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>
// CIR:   cir.store %arg0, %[[GLOBAL_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_global)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>
// CIR:   %[[GENERIC_SLOT:.*]] = cir.cast address_space %[[GENERIC_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_private)>
// CIR:   cir.store %arg1, %[[GENERIC_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_generic)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_private)>
// CIR:   %[[PRIVATE_SLOT:.*]] = cir.cast address_space %[[PRIVATE_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_private)>> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_private)>, lang_address_space(offload_private)>
// CIR:   cir.store %arg2, %[[PRIVATE_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_private)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_private)>, lang_address_space(offload_private)>

// generic_ptr = global_ptr  -->  load global_ptr slot, cast global -> generic, store into generic_ptr slot
// CIR:   %[[GLOBAL_VAL:.*]] = cir.load {{.*}} %[[GLOBAL_SLOT]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>, !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR:   %[[GLOBAL_TO_GENERIC:.*]] = cir.cast address_space %[[GLOBAL_VAL]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR:   cir.store {{.*}} %[[GLOBAL_TO_GENERIC]], %[[GENERIC_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_generic)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_private)>

// generic_ptr = private_ptr  -->  load private_ptr slot, cast private -> generic, store into generic_ptr slot
// CIR:   %[[PRIVATE_VAL:.*]] = cir.load {{.*}} %[[PRIVATE_SLOT]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_private)>, lang_address_space(offload_private)>, !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR:   %[[PRIVATE_TO_GENERIC:.*]] = cir.cast address_space %[[PRIVATE_VAL]] : !cir.ptr<!s32i, lang_address_space(offload_private)> -> !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR:   cir.store {{.*}} %[[PRIVATE_TO_GENERIC]], %[[GENERIC_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_generic)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_private)>

// global_ptr = (global int *)generic_ptr  -->  load generic_ptr slot, cast generic -> global, store into global_ptr slot
// CIR:   %[[GENERIC_VAL:.*]] = cir.load {{.*}} %[[GENERIC_SLOT]] : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_private)>, !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR:   %[[GENERIC_TO_GLOBAL:.*]] = cir.cast address_space %[[GENERIC_VAL]] : !cir.ptr<!s32i, lang_address_space(offload_generic)> -> !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR:   cir.store {{.*}} %[[GENERIC_TO_GLOBAL]], %[[GLOBAL_SLOT]] : !cir.ptr<!s32i, lang_address_space(offload_global)>, !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>
