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

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
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

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups
#error "Missing cl_intel_subgroups define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 120)
#ifndef cl_intel_subgroups_short
#error "Missing cl_intel_subgroups_short define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_intel_subgroups_short' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

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
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_int64' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_3d_image_writes : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_3d_image_writes' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_acq_rel : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_atomic_order_acq_rel' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_seq_cst : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_atomic_order_seq_cst' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_device_enqueue : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_device_enqueue' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_fp64 : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_fp64' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_generic_address_space : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_generic_address_space' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_images : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_images' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_pipes : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_pipes' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_program_scope_global_variables : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_program_scope_global_variables' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_read_write_images : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_read_write_images' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_subgroups : disable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_subgroups' - ignoring}}

#pragma OPENCL EXTENSION __opencl_c_int64 : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_int64' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_3d_image_writes : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_3d_image_writes' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_acq_rel : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_atomic_order_acq_rel' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_atomic_order_seq_cst : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_atomic_order_seq_cst' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_device_enqueue : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_device_enqueue' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_fp64 : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_fp64' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_generic_address_space : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_generic_address_space' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_images : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_images' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_pipes : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_pipes' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_program_scope_global_variables : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_program_scope_global_variables' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_read_write_images : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_read_write_images' - ignoring}}
#pragma OPENCL EXTENSION __opencl_c_subgroups : enable
//expected-warning@-1{{unknown OpenCL extension '__opencl_c_subgroups' - ignoring}}
