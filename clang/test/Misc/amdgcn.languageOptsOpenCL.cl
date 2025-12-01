// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

// RUN: %clang_cc1 -x cl -cl-std=CL3.0 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL3.0 %s -verify -triple amdgcn-unknown-unknown -target-cpu gfx700 -Wpedantic-core-features -DTEST_CORE_FEATURES -DFLAT_SUPPORT

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

#ifndef __cl_clang_non_portable_kernel_param_types
#error "Missing __cl_clang_non_portable_kernel_param_types define"
#endif
#pragma OPENCL EXTENSION __cl_clang_non_portable_kernel_param_types : enable

#ifndef __cl_clang_bitfields
#error "Missing __cl_clang_bitfields define"
#endif
#pragma OPENCL EXTENSION __cl_clang_bitfields : enable

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
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
#if (__OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_byte_addressable_store' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_global_int32_base_atomics
#error "Missing cl_khr_global_int32_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#if (__OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_global_int32_base_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_global_int32_extended_atomics
#error "Missing cl_khr_global_int32_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#if (__OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_global_int32_extended_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_local_int32_base_atomics
#error "Missing cl_khr_local_int32_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#if (__OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_local_int32_base_atomics' is core feature or supported optional core feature - ignoring}}
#endif

#ifndef cl_khr_local_int32_extended_atomics
#error "Missing cl_khr_local_int32_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#if (__OPENCL_C_VERSION__ >= 110) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_local_int32_extended_atomics' is core feature or supported optional core feature - ignoring}}
#endif

// Core feature in CL 1.2
#ifndef cl_khr_fp64
#error "Missing cl_khr_fp64 define"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#if (__OPENCL_C_VERSION__ >= 120) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_fp64' is core feature or supported optional core feature - ignoring}}
#endif

//Core feature in CL 2.0
#ifndef cl_khr_3d_image_writes
#error "Missing cl_khr_3d_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_3d_image_writes: enable
#if (__OPENCL_C_VERSION__ >= 200) && defined TEST_CORE_FEATURES
// expected-warning@-2{{OpenCL extension 'cl_khr_3d_image_writes' is core feature or supported optional core feature - ignoring}}
#endif

#ifdef cl_khr_gl_msaa_sharing
#error "Incorrect cl_khr_gl_msaa_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_msaa_sharing' - ignoring}}

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_mipmap_image
#error "Missing cl_khr_mipmap_image define"
#endif
#else
#ifdef cl_khr_mipmap_image
#error "Incorrect cl_khr_mipmap_image define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_mipmap_image' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_mipmap_image: enable

#ifdef cl_khr_srgb_image_writes
#error "Incorrect cl_khr_srgb_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_srgb_image_writes: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_srgb_image_writes' - ignoring}}

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroups
#error "Missing cl_khr_subgroups define"
#endif
#else
#ifdef cl_khr_subgroups
#error "Incorrect cl_khr_subgroups define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroups' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroups: enable

#ifndef cl_amd_media_ops
#error "Missing cl_amd_media_ops define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops: enable

#ifndef cl_amd_media_ops2
#error "Missing cl_amd_media_ops2 define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops2: enable

#if (__OPENCL_C_VERSION__ >= 300)
  #ifndef __opencl_c_program_scope_global_variables
    #error "Missing __opencl_c_program_scope_global_variables define"
  #endif
#endif

#if (__OPENCL_C_VERSION__ >= 300)
  #ifdef FLAT_SUPPORT
    #ifndef __opencl_c_generic_address_space
      #error "Missing __opencl_c_generic_address_space define"
    #endif
  #else
    #ifdef __opencl_c_generic_address_space
      #error "Incorrect __opencl_c_generic_address_space define"
    #endif
  #endif
#endif

#if (__OPENCL_C_VERSION__ >= 300)
  #ifdef FLAT_SUPPORT
    #ifndef __opencl_c_device_enqueue
      #error "Missing __opencl_c_device_enqueue define"
    #endif
  #else
    #ifdef __opencl_c_device_enqueue
      #error "Incorrect __opencl_c_device_enqueue define"
    #endif
  #endif
#endif
