// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple nvptx-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple nvptx-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple nvptx-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple nvptx-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple nvptx-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple nvptx-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple nvptx-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple nvptx-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple nvptx64-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple nvptx64-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple nvptx64-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple nvptx64-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple nvptx64-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple nvptx64-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple nvptx64-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple nvptx64-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

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

#ifdef cl_khr_fp16
#error "Incorrect cl_khr_fp16 define"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_fp16' - ignoring}}

#ifdef cl_khr_int64_base_atomics
#error "Incorrect cl_khr_int64_base_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_int64_base_atomics' - ignoring}}

#ifdef cl_khr_int64_extended_atomics
#error "Incorrect cl_khr_int64_extended_atomics define"
#endif
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_int64_extended_atomics' - ignoring}}

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

// Core feature in CL 2.0, but not supported on nvptx
#ifdef cl_khr_3d_image_writes
#error "Incorrect cl_khr_3d_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_3d_image_writes: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_3d_image_writes' - ignoring}}

#ifdef cl_khr_gl_msaa_sharing
#error "Incorrect cl_khr_gl_msaa_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_msaa_sharing' - ignoring}}

#ifdef cl_khr_srgb_image_writes
#error "Incorrect cl_khr_srgb_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_srgb_image_writes: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_srgb_image_writes' - ignoring}}

#ifdef cl_khr_subgroups
#error "Incorrect cl_khr_subgroups define"
#endif
#pragma OPENCL EXTENSION cl_khr_subgroups: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_subgroups' - ignoring}}

