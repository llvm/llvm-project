// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=clc++ %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL3.0 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=clc++ %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL3.0 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

// Extensions in all versions
#ifndef cl_clang_storage_class_specifiers
#error "Missing cl_clang_storage_class_specifiers define"
#endif
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#ifndef __cl_clang_function_pointers
#error "Missing __cl_clang_function_pointers define"
#endif
#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable

#ifndef __cl_clang_variadic_functions
#error "Missing __cl_clang_variadic_functions define"
#endif
#pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable

#ifndef cl_khr_fp16
#error "Missing cl_khr_fp16 define"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16: enable

#ifndef cl_khr_int64_base_atomics
#error "Missing cl_khr_int64_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#ifndef cl_khr_int64_extended_atomics
#error "Missing cl_khr_int64_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

// Core features in CL 1.1

#ifndef cl_khr_byte_addressable_store
#error "Missing cl_khr_byte_addressable_store define"
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_byte_addressable_store' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_global_int32_base_atomics
#error "Missing cl_khr_global_int32_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_global_int32_base_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_global_int32_extended_atomics
#error "Missing cl_khr_global_int32_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_global_int32_extended_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_local_int32_base_atomics
#error "Missing cl_khr_local_int32_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_local_int32_base_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_local_int32_extended_atomics
#error "Missing cl_khr_local_int32_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_local_int32_extended_atomics' is core feature or supported optional core feature - ignoring}}
#endif

// Core feature in CL 1.2
#ifndef cl_khr_fp64
#error "Missing cl_khr_fp64 define"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_fp64' is core feature or supported optional core feature - ignoring}}
#endif

//Core feature in CL 2.0, optional core feature in CL 3.0
#ifndef cl_khr_3d_image_writes
#error "Missing cl_khr_3d_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ == 200 || __OPENCL_C_VERSION__ == 300) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_3d_image_writes' is core feature or supported optional core feature - ignoring}}
#endif

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 110)
#ifndef cles_khr_int64
#error "Missing cles_khr_int64 define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cles_khr_int64' - ignoring}}
#endif
#pragma OPENCL EXTENSION cles_khr_int64 : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 100)
#ifndef cl_khr_gl_msaa_sharing
#error "Missing cl_khr_gl_msaa_sharing define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_gl_msaa_sharing' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_mipmap_image
#error "Missing cl_khr_mipmap_image define"
#endif
#else
#ifdef cl_khr_mipmap_image
#error "Incorrect cl_khr_mipmap_image define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_mipmap_image' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_mipmap_image : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_mipmap_image_writes
#error "Missing cl_khr_mipmap_image_writes define"
#endif
#else
#ifdef cl_khr_mipmap_image_writes
#error "Incorrect cl_khr_mipmap_image_writes define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_mipmap_image_writes' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_mipmap_image_writes : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_srgb_image_writes
#error "Missing cl_khr_srgb_image_writes define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_srgb_image_writes' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_srgb_image_writes : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroups
#error "Missing cl_khr_subgroups define"
#endif
#else
#ifdef cl_khr_subgroups
#error "Incorrect cl_khr_subgroups define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroups' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_ext_float_atomics
#error "Missing cl_ext_float_atomics define"
#endif
#else
#ifdef cl_ext_float_atomics
#error "Incorrect cl_ext_float_atomics define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_ext_float_atomics' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_extended_bit_ops
#error "Missing cl_khr_extended_bit_ops define"
#endif
#else
#ifdef cl_khr_extended_bit_ops
#error "Incorrect cl_khr_extended_bit_ops define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_extended_bit_ops' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_extended_bit_ops : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_integer_dot_product
#error "Missing cl_khr_integer_dot_product define"
#endif
#else
#ifdef cl_khr_integer_dot_product
#error "Incorrect cl_khr_integer_dot_product define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_integer_dot_product' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_kernel_clock
#error "Missing cl_khr_kernel_clock define"
#endif
#else
#ifdef cl_khr_kernel_clock
#error "Incorrect cl_khr_kernel_clock define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_kernel_clock' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_kernel_clock : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_ballot
#error "Missing cl_khr_subgroup_ballot define"
#endif
#else
#ifdef cl_khr_subgroup_ballot
#error "Incorrect cl_khr_subgroup_ballot define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_ballot' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_clustered_reduce
#error "Missing cl_khr_subgroup_clustered_reduce define"
#endif
#else
#ifdef cl_khr_subgroup_clustered_reduce
#error "Incorrect cl_khr_subgroup_clustered_reduce define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_clustered_reduce' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_clustered_reduce : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_extended_types
#error "Missing cl_khr_subgroup_extended_types define"
#endif
#else
#ifdef cl_khr_subgroup_extended_types
#error "Incorrect cl_khr_subgroup_extended_types define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_extended_types' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_named_barrier
#error "Missing cl_khr_subgroup_named_barrier define"
#endif
#else
#ifdef cl_khr_subgroup_named_barrier
#error "Incorrect cl_khr_subgroup_named_barrier define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_named_barrier' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_named_barrier : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_non_uniform_arithmetic
#error "Missing cl_khr_subgroup_non_uniform_arithmetic define"
#endif
#else
#ifdef cl_khr_subgroup_non_uniform_arithmetic
#error "Incorrect cl_khr_subgroup_non_uniform_arithmetic define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_non_uniform_arithmetic' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_arithmetic : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_non_uniform_vote
#error "Missing cl_khr_subgroup_non_uniform_vote define"
#endif
#else
#ifdef cl_khr_subgroup_non_uniform_vote
#error "Incorrect cl_khr_subgroup_non_uniform_vote define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_non_uniform_vote' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_rotate
#error "Missing cl_khr_subgroup_rotate define"
#endif
#else
#ifdef cl_khr_subgroup_rotate
#error "Incorrect cl_khr_subgroup_rotate define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_rotate' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_rotate : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_shuffle_relative
#error "Missing cl_khr_subgroup_shuffle_relative define"
#endif
#else
#ifdef cl_khr_subgroup_shuffle_relative
#error "Incorrect cl_khr_subgroup_shuffle_relative define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_shuffle_relative' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_shuffle
#error "Missing cl_khr_subgroup_shuffle define"
#endif
#else
#ifdef cl_khr_subgroup_shuffle
#error "Incorrect cl_khr_subgroup_shuffle define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_khr_subgroup_shuffle' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable

#ifndef cl_amd_media_ops
#error "Missing cl_amd_media_ops define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops: enable

#ifndef cl_amd_media_ops2
#error "Missing cl_amd_media_ops2 define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_depth_images
#error "Missing cl_khr_depth_images define"
#endif
#else
#ifdef cl_khr_depth_images
#error "Incorrect cl_khr_depth_images define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_depth_images' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_depth_images : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 100)
#ifndef cl_intel_bfloat16_conversions
#error "Missing cl_intel_bfloat16_conversions define"
#endif
#else
#ifdef cl_intel_bfloat16_conversions
#error "Incorrect cl_intel_bfloat16_conversions define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_intel_bfloat16_conversions' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_intel_required_subgroup_size
#error "Missing cl_intel_required_subgroup_size define"
#endif
#else
#ifdef cl_intel_required_subgroup_size
#error "Incorrect cl_intel_required_subgroup_size define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_intel_required_subgroup_size' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups
#error "Missing cl_intel_subgroups define"
#endif
#else
#ifdef cl_intel_subgroups
#error "Incorrect cl_intel_subgroups define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups_char
#error "Missing cl_intel_subgroups_char define"
#endif
#else
#ifdef cl_intel_subgroups_char
#error "Incorrect cl_intel_subgroups_char define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups_char' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups_long
#error "Missing cl_intel_subgroups_long define"
#endif
#else
#ifdef cl_intel_subgroups_long
#error "Incorrect cl_intel_subgroups_long define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups_long' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups_long : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups_short
#error "Missing cl_intel_subgroups_short define"
#endif
#else
#ifdef cl_intel_subgroups_short
#error "Incorrect cl_intel_subgroups_short define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups_short' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroup_buffer_prefetch
#error "Missing cl_intel_subgroup_buffer_prefetch define"
#endif
#else
#ifdef cl_intel_subgroup_buffer_prefetch
#error "Incorrect cl_intel_subgroup_buffer_prefetch define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_intel_subgroup_buffer_prefetch' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_intel_subgroup_buffer_prefetch : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroup_local_block_io
#error "Missing cl_intel_subgroup_local_block_io define"
#endif
#else
#ifdef cl_intel_subgroup_local_block_io
#error "Incorrect cl_intel_subgroup_local_block_io define"
#endif
#endif
// expected-warning@+1{{OpenCL extension 'cl_intel_subgroup_local_block_io' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_device_side_avc_motion_estimation
#error "Missing cl_intel_device_side_avc_motion_estimation define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_device_side_avc_motion_estimation' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable

// Check that pragmas for the OpenCL 3.0 features are rejected.

#pragma OPENCL EXTENSION __opencl_c_int64 : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_int64' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_3d_image_writes : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_3d_image_writes' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_acq_rel : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_order_acq_rel' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_seq_cst : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_order_seq_cst' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_scope_all_devices : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_scope_all_devices' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_scope_device : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_scope_device' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_device_enqueue : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_device_enqueue' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_load_store : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_load_store' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_load_store : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_load_store' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_global_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_global_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_local_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_local_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_global_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_global_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_local_atomic_add : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_local_atomic_min_max : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_raw10_raw12 : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_raw10_raw12' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_unorm_int_2_101010 : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_unorm_int_2_101010' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_unsigned_10x6_12x4_14x2 : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_unsigned_10x6_12x4_14x2' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_fp64 : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_fp64' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_generic_address_space : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_generic_address_space' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_images : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_images' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_integer_dot_product_input_4x8bit : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_integer_dot_product_input_4x8bit' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_integer_dot_product_input_4x8bit_packed : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_integer_dot_product_input_4x8bit_packed' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_device : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_device' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_sub_group : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_sub_group' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_work_group : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_work_group' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_pipes : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_pipes' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_program_scope_global_variables : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_program_scope_global_variables' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_read_write_images : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_read_write_images' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_subgroups : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_subgroups' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_work_group_collective_functions : disable
//expected-warning@-1{{OpenCL extension '__opencl_c_work_group_collective_functions' unknown or does not require pragma - ignoring}}

#pragma OPENCL EXTENSION __opencl_c_int64 : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_int64' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_3d_image_writes : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_3d_image_writes' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_acq_rel : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_order_acq_rel' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_seq_cst : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_order_seq_cst' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_scope_all_devices : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_scope_all_devices' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_scope_device : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_atomic_scope_device' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_device_enqueue : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_device_enqueue' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_load_store : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_load_store' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_global_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_load_store : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_load_store' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp16_local_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp16_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_global_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_global_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_local_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp32_local_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp32_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_global_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_global_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_global_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_global_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_local_atomic_add : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_local_atomic_add' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_fp64_local_atomic_min_max : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_fp64_local_atomic_min_max' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_raw10_raw12 : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_raw10_raw12' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_unorm_int_2_101010 : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_unorm_int_2_101010' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_ext_image_unsigned_10x6_12x4_14x2 : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_ext_image_unsigned_10x6_12x4_14x2' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_fp64 : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_fp64' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_generic_address_space : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_generic_address_space' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_images : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_images' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_integer_dot_product_input_4x8bit : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_integer_dot_product_input_4x8bit' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_integer_dot_product_input_4x8bit_packed : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_integer_dot_product_input_4x8bit_packed' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_device : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_device' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_sub_group : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_sub_group' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_kernel_clock_scope_work_group : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_kernel_clock_scope_work_group' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_pipes : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_pipes' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_program_scope_global_variables : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_program_scope_global_variables' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_read_write_images : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_read_write_images' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_subgroups : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_subgroups' unknown or does not require pragma - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_work_group_collective_functions : enable
//expected-warning@-1{{OpenCL extension '__opencl_c_work_group_collective_functions' unknown or does not require pragma - ignoring}}
