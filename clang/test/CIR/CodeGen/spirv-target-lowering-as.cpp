// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-cir \
// RUN:   -Wno-deprecated-attributes -mmlir \
// RUN:   --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=PRE --input-file=%t.pre.cir
// RUN: FileCheck %s --check-prefix=POST --input-file=%t.cir

using global_int = int [[clang::opencl_global]];
using local_int = int [[clang::opencl_local]];
using constant_int = int [[clang::opencl_constant]];
using private_int = int [[clang::opencl_private]];
using generic_int = int [[clang::opencl_generic]];
using device_int = int [[clang::opencl_global_device]];
using host_int = int [[clang::opencl_global_host]];

global_int global_value = 0;

// PRE: cir.global {{.*}} lang_address_space(offload_global) @global_value
// POST: cir.global {{.*}} target_address_space(1) @global_value

void address_spaces(private_int *private_ptr, local_int *local_ptr,
                    global_int *global_ptr, constant_int *constant_ptr,
                    generic_int *generic_ptr, device_int *device_ptr,
                    host_int *host_ptr) {}

// PRE-LABEL: cir.func {{.*}} @_Z14address_spaces
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_private)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_local)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_constant)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_global_device)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_global_host)>

// POST-LABEL: cir.func {{.*}} @_Z14address_spaces
// POST-SAME: !cir.ptr<!s32i>
// POST-SAME: !cir.ptr<!s32i, target_address_space(3)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(1)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(2)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(4)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(5)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(6)>

using global_int_ptr = global_int *;
using global_int_ptr_array = global_int_ptr[4];

void nested_types(global_int_ptr [[clang::opencl_local]] *nested,
                  global_int_ptr_array [[clang::opencl_constant]] *array) {}

// PRE-LABEL: cir.func {{.*}} @_Z12nested_types
// PRE-SAME: !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_local)>
// PRE-SAME: !cir.ptr<!cir.array<!cir.ptr<!s32i, lang_address_space(offload_global)> x 4>, lang_address_space(offload_constant)>

// POST-LABEL: cir.func {{.*}} @_Z12nested_types
// POST-SAME: !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>, target_address_space(3)>
// POST-SAME: !cir.ptr<!cir.array<!cir.ptr<!s32i, target_address_space(1)> x 4>, target_address_space(2)>

generic_int *cast_and_global(global_int *ptr) {
  (void)global_value;
  return ptr;
}

// PRE-LABEL: cir.func {{.*}} @_Z15cast_and_global
// PRE: cir.get_global @global_value : !cir.ptr<!s32i, lang_address_space(offload_global)>
// PRE: cir.cast address_space
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_generic)>

// POST-LABEL: cir.func {{.*}} @_Z15cast_and_global
// POST: cir.get_global @global_value : !cir.ptr<!s32i, target_address_space(1)>
// POST: cir.cast address_space
// POST-SAME: !cir.ptr<!s32i, target_address_space(1)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(4)>
