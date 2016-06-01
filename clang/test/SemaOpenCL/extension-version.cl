// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

#if __OPENCL_C_VERSION__ >= 200 && ! defined TEST_CORE_FEATURES
// expected-no-diagnostics
#endif

// Extensions in all versions
#ifndef cl_clang_storage_class_specifiers
#error "Missing cl_clang_storage_class_specifiers define"
#endif
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers: enable

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

#ifndef cl_khr_gl_sharing
#error "Missing cl_khr_gl_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_sharing: enable

#ifndef cl_khr_icd
#error "Missing cl_khr_icd define"
#endif
#pragma OPENCL EXTENSION cl_khr_icd: enable

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

#if (__OPENCL_C_VERSION__ < 110)
// Deprecated abvoe 1.0
#ifndef cl_khr_select_fprounding_mode
#error "Missing cl_khr_select_fp_rounding_mode define"
#endif
#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode: enable
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



#if (__OPENCL_C_VERSION__ >= 110)
#ifndef cl_khr_gl_event
#error "Missing cl_khr_gl_event define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_gl_event' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_gl_event: enable

#if (__OPENCL_C_VERSION__ >= 110)
#ifndef cl_khr_d3d10_sharing
#error "Missing cl_khr_d3d10_sharing define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_d3d10_sharing' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_d3d10_sharing: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_context_abort
#error "Missing cl_context_abort define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_context_abort' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_context_abort: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_d3d11_sharing
#error "Missing cl_khr_d3d11_sharing define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_d3d11_sharing' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_d3d11_sharing: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_dx9_media_sharing
#error "Missing cl_khr_dx9_media_sharing define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_dx9_media_sharing' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_dx9_media_sharing: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_image2d_from_buffer
#error "Missing cl_khr_image2d_from_buffer define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_image2d_from_buffer' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_image2d_from_buffer: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_initialize_memory
#error "Missing cl_khr_initialize_memory define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_initialize_memory' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_initialize_memory: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_gl_depth_images
#error "Missing cl_khr_gl_depth_images define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_gl_depth_images' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_gl_depth_images: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_gl_msaa_sharing
#error "Missing cl_khr_gl_msaa_sharing define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_gl_msaa_sharing' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing: enable

#if (__OPENCL_C_VERSION__ >= 120)
#ifndef cl_khr_spir
#error "Missing cl_khr_spir define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_spir' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_spir: enable

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_egl_event
#error "Missing cl_khr_egl_event define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_egl_event' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_egl_event: enable

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_egl_image
#error "Missing cl_khr_egl_image define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_egl_image' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_egl_image: enable

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_srgb_image_writes
#error "Missing cl_khr_srgb_image_writes define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_srgb_image_writes' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_srgb_image_writes: enable

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroups
#error "Missing cl_khr_subgroups define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroups' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroups: enable

#if (__OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_terminate_context
#error "Missing cl_khr_terminate_context define"
#endif
#else
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_terminate_context' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_terminate_context: enable
