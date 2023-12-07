// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=CL1.1 | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=CL1.2 | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=clc++1.0 | FileCheck %s --check-prefix=CHECK20
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=CL3.0 | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=clc++2021 | FileCheck %s

// RUN: %clang_cc1 -O0 -triple spirv32-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify | FileCheck %s


// Test including the default header as a module.
// The module should be compiled only once and loaded from cache afterwards.
// Change the directory mode to read only to make sure no new modules are created.
// Check time report to make sure module is used.
// Check that some builtins occur in the generated IR when called.

// ===
// Clear current directory.
// RUN: rm -rf %t
// RUN: mkdir -p %t

// ===
// Compile for OpenCL 1.0 for the first time. A module should be generated.
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -o - -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: chmod u-w %t/opencl_c.pcm

// ===
// Compile for OpenCL 1.0 for the second time. The module should not be re-created.
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -o - -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: chmod u+w %t/opencl_c.pcm
// RUN: mv %t/opencl_c.pcm %t/1_0.pcm

// ===
// Compile for OpenCL 2.0 for the first time. The module should change.
// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: not diff %t/1_0.pcm %t/opencl_c.pcm
// RUN: chmod u-w %t/opencl_c.pcm

// ===
// Compile for OpenCL 2.0 for the second time. The module should not change.
// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s

// Check cached module works for different OpenCL versions.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm -o - -cl-std=CL1.2 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: %clang_cc1 -triple amdgcn--amdhsa -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: chmod u-w %t 
// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm -o - -cl-std=CL1.2 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: %clang_cc1 -triple amdgcn--amdhsa -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: chmod u+w %t

// Verify that called builtins occur in the generated IR.

// CHECK-NOT: intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty
// CHECK-NOT: ndrange_t
// CHECK20: ndrange_t
// CHECK: _Z16convert_char_rtec
// CHECK-NOT: _Z3ctzc
// CHECK20: _Z3ctzc
// CHECK20: _Z16convert_char_rtec
char f(char x) {
// Check functionality from OpenCL 2.0 onwards
#if (__OPENCL_CPP_VERSION__ == 100) || (__OPENCL_C_VERSION__ == CL_VERSION_2_0)
  ndrange_t t;
  x = ctz(x);
#endif //__OPENCL_C_VERSION__
  return convert_char_rte(x);
}

// Verify that a builtin using a write_only image3d_t type is available
// from OpenCL 2.0 onwards.

// CHECK20: _Z12write_imagef14ocl_image3d_wo
#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= CL_VERSION_2_0)
void test_image3dwo(write_only image3d_t img) {
  write_imagef(img, (0), (0.0f));
}
#endif //__OPENCL_C_VERSION__

#if defined(__OPENCL_CPP_VERSION__)
// Test old atomic overloaded with generic addr space.
void test_atomics(__generic volatile unsigned int* a) {
  atomic_add(a, 1);
}
#endif

// Verify that ATOMIC_VAR_INIT is defined.
#if (__OPENCL_CPP_VERSION__ == 100) || (__OPENCL_C_VERSION__ == CL_VERSION_2_0)
global atomic_int z = ATOMIC_VAR_INIT(99);
#endif //__OPENCL_C_VERSION__
// CHECK-MOD: Reading modules

// Check that extension macros are defined correctly.

// For SPIR all extensions are supported.
#if defined(__SPIR__) || defined(__SPIRV__)

// Verify that cl_intel_planar_yuv extension is defined from OpenCL 1.2 onwards.
#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= CL_VERSION_1_2)
// expected-no-diagnostics
#else //__OPENCL_C_VERSION__
// expected-warning@+2{{OpenCL extension 'cl_intel_planar_yuv' unknown or does not require pragma - ignoring}}
#endif //__OPENCL_C_VERSION__
#pragma OPENCL EXTENSION cl_intel_planar_yuv : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)

#if cl_khr_subgroup_extended_types != 1
#error "Incorrectly defined cl_khr_subgroup_extended_types"
#endif
#if cl_khr_subgroup_non_uniform_vote != 1
#error "Incorrectly defined cl_khr_subgroup_non_uniform_vote"
#endif
#if cl_khr_subgroup_ballot != 1
#error "Incorrectly defined cl_khr_subgroup_ballot"
#endif
#if cl_khr_subgroup_non_uniform_arithmetic != 1
#error "Incorrectly defined cl_khr_subgroup_non_uniform_arithmetic"
#endif
#if cl_khr_subgroup_shuffle != 1
#error "Incorrectly defined cl_khr_subgroup_shuffle"
#endif
#if cl_khr_subgroup_shuffle_relative != 1
#error "Incorrectly defined cl_khr_subgroup_shuffle_relative"
#endif
#if cl_khr_subgroup_clustered_reduce != 1
#error "Incorrectly defined cl_khr_subgroup_clustered_reduce"
#endif
#if cl_khr_subgroup_rotate != 1
#error "Incorrectly defined cl_khr_subgroup_rotate"
#endif
#if cl_khr_extended_bit_ops != 1
#error "Incorrectly defined cl_khr_extended_bit_ops"
#endif
#if cl_khr_integer_dot_product != 1
#error "Incorrectly defined cl_khr_integer_dot_product"
#endif
#if __opencl_c_integer_dot_product_input_4x8bit != 1
#error "Incorrectly defined __opencl_c_integer_dot_product_input_4x8bit"
#endif
#if __opencl_c_integer_dot_product_input_4x8bit_packed != 1
#error "Incorrectly defined __opencl_c_integer_dot_product_input_4x8bit_packed"
#endif
#if cl_ext_float_atomics != 1
#error "Incorrectly defined cl_ext_float_atomics"
#endif
#if __opencl_c_ext_fp16_global_atomic_load_store != 1
#error "Incorrectly defined __opencl_c_ext_fp16_global_atomic_load_store"
#endif
#if __opencl_c_ext_fp16_local_atomic_load_store != 1
#error "Incorrectly defined __opencl_c_ext_fp16_local_atomic_load_store"
#endif
#if __opencl_c_ext_fp16_global_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp16_global_atomic_add"
#endif
#if __opencl_c_ext_fp32_global_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp32_global_atomic_add"
#endif
#if __opencl_c_ext_fp64_global_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp64_global_atomic_add"
#endif
#if __opencl_c_ext_fp16_local_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp16_local_atomic_add"
#endif
#if __opencl_c_ext_fp32_local_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp32_local_atomic_add"
#endif
#if __opencl_c_ext_fp64_local_atomic_add != 1
#error "Incorrectly defined __opencl_c_ext_fp64_local_atomic_add"
#endif
#if __opencl_c_ext_fp16_global_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp16_global_atomic_min_max"
#endif
#if __opencl_c_ext_fp32_global_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp32_global_atomic_min_max"
#endif
#if __opencl_c_ext_fp64_global_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp64_global_atomic_min_max"
#endif
#if __opencl_c_ext_fp16_local_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp16_local_atomic_min_max"
#endif
#if __opencl_c_ext_fp32_local_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp32_local_atomic_min_max"
#endif
#if __opencl_c_ext_fp64_local_atomic_min_max != 1
#error "Incorrectly defined __opencl_c_ext_fp64_local_atomic_min_max"
#endif
#if __opencl_c_ext_image_raw10_raw12 != 1
#error "Incorrectly defined __opencl_c_ext_image_raw10_raw12"
#endif

#else

#ifdef cl_khr_subgroup_extended_types
#error "Incorrect cl_khr_subgroup_extended_types define"
#endif
#ifdef cl_khr_subgroup_non_uniform_vote
#error "Incorrect cl_khr_subgroup_non_uniform_vote define"
#endif
#ifdef cl_khr_subgroup_ballot
#error "Incorrect cl_khr_subgroup_ballot define"
#endif
#ifdef cl_khr_subgroup_non_uniform_arithmetic
#error "Incorrect cl_khr_subgroup_non_uniform_arithmetic define"
#endif
#ifdef cl_khr_subgroup_shuffle
#error "Incorrect cl_khr_subgroup_shuffle define"
#endif
#ifdef cl_khr_subgroup_shuffle_relative
#error "Incorrect cl_khr_subgroup_shuffle_relative define"
#endif
#ifdef cl_khr_subgroup_clustered_reduce
#error "Incorrect cl_khr_subgroup_clustered_reduce define"
#endif
#ifdef cl_khr_subgroup_rotate
#error "Incorrect cl_khr_subgroup_rotate define"
#endif
#ifdef cl_khr_extended_bit_ops
#error "Incorrect cl_khr_extended_bit_ops define"
#endif
#ifdef cl_khr_integer_dot_product
#error "Incorrect cl_khr_integer_dot_product define"
#endif
#ifdef __opencl_c_integer_dot_product_input_4x8bit
#error "Incorrect __opencl_c_integer_dot_product_input_4x8bit define"
#endif
#ifdef __opencl_c_integer_dot_product_input_4x8bit_packed
#error "Incorrect __opencl_c_integer_dot_product_input_4x8bit_packed define"
#endif
#ifdef cl_ext_float_atomics
#error "Incorrect cl_ext_float_atomics define"
#endif
#ifdef __opencl_c_ext_fp16_global_atomic_load_store
#error "Incorrect __opencl_c_ext_fp16_global_atomic_load_store define"
#endif
#ifdef __opencl_c_ext_fp16_local_atomic_load_store
#error "Incorrect __opencl_c_ext_fp16_local_atomic_load_store define"
#endif
#ifdef __opencl_c_ext_fp16_global_atomic_add
#error "Incorrect __opencl_c_ext_fp16_global_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp32_global_atomic_add
#error "Incorrect __opencl_c_ext_fp32_global_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp64_global_atomic_add
#error "Incorrect __opencl_c_ext_fp64_global_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp16_local_atomic_add
#error "Incorrect __opencl_c_ext_fp16_local_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp32_local_atomic_add
#error "Incorrect __opencl_c_ext_fp32_local_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp64_local_atomic_add
#error "Incorrect __opencl_c_ext_fp64_local_atomic_add define"
#endif
#ifdef __opencl_c_ext_fp16_global_atomic_min_max
#error "Incorrect __opencl_c_ext_fp16_global_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_fp32_global_atomic_min_max
#error "Incorrect __opencl_c_ext_fp32_global_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_fp64_global_atomic_min_max
#error "Incorrect __opencl_c_ext_fp64_global_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_fp16_local_atomic_min_max
#error "Incorrect __opencl_c_ext_fp16_local_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_fp32_local_atomic_min_max
#error "Incorrect __opencl_c_ext_fp32_local_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_fp64_local_atomic_min_max
#error "Incorrect __opencl_c_ext_fp64_local_atomic_min_max define"
#endif
#ifdef __opencl_c_ext_image_raw10_raw12
#error "Incorrect __opencl_c_ext_image_raw10_raw12 define"
#endif

#endif //(defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)

// OpenCL C features.
#if (__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300)

#if __opencl_c_atomic_scope_all_devices != 1
#error "Incorrectly defined feature macro __opencl_c_atomic_scope_all_devices"
#endif

#elif (__OPENCL_CPP_VERSION__ == 100 || __OPENCL_C_VERSION__ == 200)

#ifndef  __opencl_c_pipes
#error "Feature macro __opencl_c_pipes should be defined"
#endif
#ifndef __opencl_c_generic_address_space
#error "Feature macro __opencl_c_generic_address_space should be defined"
#endif
#ifndef __opencl_c_work_group_collective_functions
#error "Feature macro __opencl_c_work_group_collective_functions should be defined"
#endif
#ifndef __opencl_c_atomic_order_acq_rel
#error "Feature macro __opencl_c_atomic_order_acq_rel should be defined"
#endif
#ifndef __opencl_c_atomic_order_seq_cst
#error "Feature macro __opencl_c_atomic_order_seq_cst should be defined"
#endif
#ifndef __opencl_c_atomic_scope_device
#error "Feature macro __opencl_c_atomic_scope_device should be defined"
#endif
#ifndef __opencl_c_atomic_scope_all_devices
#error "Feature macro __opencl_c_atomic_scope_all_devices should be defined"
#endif
#ifndef __opencl_c_device_enqueue
#error "Feature macro __opencl_c_device_enqueue should be defined"
#endif
#ifndef __opencl_c_read_write_images
#error "Feature macro __opencl_c_read_write_images should be defined"
#endif
#ifndef __opencl_c_program_scope_global_variables
#error "Feature macro __opencl_c_program_scope_global_variables should be defined"
#endif
#ifndef __opencl_c_images
#error "Feature macro __opencl_c_images should be defined"
#endif

#elif (__OPENCL_C_VERSION__ < 200)

#ifdef  __opencl_c_pipes
#error "Incorrect feature macro __opencl_c_pipes define"
#endif
#ifdef __opencl_c_generic_address_space
#error "Incorrect feature macro __opencl_c_generic_address_space define"
#endif
#ifdef __opencl_c_work_group_collective_functions
#error "Incorrect feature macro __opencl_c_work_group_collective_functions define"
#endif
#ifdef __opencl_c_atomic_order_acq_rel
#error "Incorrect feature macro __opencl_c_atomic_order_acq_rel define"
#endif
#ifdef __opencl_c_atomic_order_seq_cst
#error "Incorrect feature macro __opencl_c_atomic_order_seq_cst define"
#endif
#ifdef __opencl_c_atomic_scope_device
#error "Incorrect feature macro __opencl_c_atomic_scope_device define"
#endif
#ifdef __opencl_c_atomic_scope_all_devices
#error "Incorrect feature macro __opencl_c_atomic_scope_all_devices define"
#endif
#ifdef __opencl_c_device_enqueue
#error "Incorrect feature macro __opencl_c_device_enqueue define"
#endif
#ifdef __opencl_c_read_write_images
#error "Incorrect feature macro __opencl_c_read_write_images define"
#endif
#ifdef __opencl_c_program_scope_global_variables
#error "Incorrect feature macro __opencl_c_program_scope_global_variables define"
#endif
#ifdef __opencl_c_images
#error "Incorrect feature macro __opencl_c_images define"
#endif
#ifdef __opencl_c_3d_image_writes
#error "Incorrect feature macro __opencl_c_3d_image_writes define"
#endif
#ifdef __opencl_c_fp64
#error "Incorrect feature macro __opencl_c_fp64 define"
#endif
#ifdef __opencl_c_subgroups
#error "Incorrect feature macro __opencl_c_subgroups define"
#endif

#endif // (__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300)

#endif // defined(__SPIR__)
