// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -fclangir -emit-cir \
// RUN:   -Wno-deprecated-attributes -mmlir \
// RUN:   --mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir

typedef global int *global_int_ptr;

void pointer_types(
    private int *private_ptr, local int *local_ptr, global int *global_ptr,
    constant int *constant_ptr, generic int *generic_ptr,
    __attribute__((opencl_global_device)) int *global_device_ptr,
    __attribute__((opencl_global_host)) int *global_host_ptr) {}

// CIR-LABEL: cir.func dso_local @pointer_types
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_local)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_constant)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global_device)>
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global_host)>

// The outer pointer addresses a private pointer object, whose stored pointer
// value addresses global memory.
void nested_pointer(global_int_ptr private *ptr) {}

// CIR-LABEL: cir.func dso_local @nested_pointer
// CIR-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>

void local_pointer_value(global int *ptr) {
  global int *saved = ptr;
  (void)saved;
}

// CIR-LABEL: cir.func dso_local @local_pointer_value
// CIR: %[[SAVED:.*]] = cir.alloca "saved"
// CIR-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>>
// CIR: %[[SAVED_ADDR:.*]] = cir.cast address_space %[[SAVED]]
// CIR-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>>
// CIR-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>
// CIR: cir.store {{.*}}, %[[SAVED_ADDR]]
// CIR-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>
