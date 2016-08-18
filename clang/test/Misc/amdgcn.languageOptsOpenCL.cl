// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple amdgcn-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple amdgcn-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

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

#ifdef cl_khr_gl_sharing
#error "Incorrect cl_khr_gl_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_sharing' - ignoring}}

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

#ifdef cl_khr_select_fprounding_mode
#error "Incorrect cl_khr_select_fprounding_mode define"
#endif
#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_select_fprounding_mode' - ignoring}}


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



#ifdef cl_khr_gl_event
#error "Incorrect cl_khr_gl_event define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_event: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_event' - ignoring}}

#ifdef cl_khr_d3d10_sharing
#error "Incorrect cl_khr_d3d10_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_d3d10_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_d3d10_sharing' - ignoring}}

#ifdef cl_khr_context_abort
#error "Incorrect cl_context_abort define"
#endif
#pragma OPENCL EXTENSION cl_khr_context_abort: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_context_abort' - ignoring}}

#ifdef cl_khr_d3d11_sharing
#error "Incorrect cl_khr_d3d11_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_d3d11_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_d3d11_sharing' - ignoring}}

#ifdef cl_khr_dx9_media_sharing
#error "Incorrect cl_khr_dx9_media_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_dx9_media_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_dx9_media_sharing' - ignoring}}

#ifdef cl_khr_image2d_from_buffer
#error "Incorrect cl_khr_image2d_from_buffer define"
#endif
#pragma OPENCL EXTENSION cl_khr_image2d_from_buffer: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_image2d_from_buffer' - ignoring}}

#ifdef cl_khr_initialize_memory
#error "Incorrect cl_khr_initialize_memory define"
#endif
#pragma OPENCL EXTENSION cl_khr_initialize_memory: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_initialize_memory' - ignoring}}

#ifdef cl_khr_gl_depth_images
#error "Incorrect cl_khr_gl_depth_images define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_depth_images: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_depth_images' - ignoring}}

#ifdef cl_khr_gl_msaa_sharing
#error "Incorrect cl_khr_gl_msaa_sharing define"
#endif
#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_gl_msaa_sharing' - ignoring}}

#ifdef cl_khr_spir
#error "Incorrect cl_khr_spir define"
#endif
#pragma OPENCL EXTENSION cl_khr_spir: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_spir' - ignoring}}

#ifdef cl_khr_egl_event
#error "Incorrect cl_khr_egl_event define"
#endif
#pragma OPENCL EXTENSION cl_khr_egl_event: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_egl_event' - ignoring}}

#ifdef cl_khr_egl_image
#error "Incorrect cl_khr_egl_image define"
#endif
#pragma OPENCL EXTENSION cl_khr_egl_image: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_egl_image' - ignoring}}

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

#ifdef cl_khr_terminate_context
#error "Incorrect cl_khr_terminate_context define"
#endif
#pragma OPENCL EXTENSION cl_khr_terminate_context: enable
// expected-warning@-1{{unsupported OpenCL extension 'cl_khr_terminate_context' - ignoring}}

#ifndef cl_amd_media_ops
#error "Missing cl_amd_media_ops define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops: enable

#ifndef cl_amd_media_ops2
#error "Missing cl_amd_media_ops2 define"
#endif
#pragma OPENCL EXTENSION cl_amd_media_ops2: enable

