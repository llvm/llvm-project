// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=clc++ %s -verify -triple spir-unknown-unknown
// RUN: %clang_cc1 -x cl -cl-std=CL %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES
// RUN: %clang_cc1 -x cl -cl-std=clc++ %s -verify -triple spir-unknown-unknown -Wpedantic-core-features -DTEST_CORE_FEATURES

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200) && !defined(TEST_CORE_FEATURES)
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

//Core feature in CL 2.0
#ifndef cl_khr_3d_image_writes
#error "Missing cl_khr_3d_image_writes define"
#endif
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200) && defined TEST_CORE_FEATURES
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

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_extended_types
#error "Missing cl_khr_subgroup_extended_types"
#endif
#else
#ifdef cl_khr_subgroup_extended_types
#error "Incorrect cl_khr_subgroup_extended_types define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_extended_types' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_non_uniform_vote
#error "Missing cl_khr_subgroup_non_uniform_vote"
#endif
#else
#ifdef cl_khr_subgroup_non_uniform_vote
#error "Incorrect cl_khr_subgroup_non_uniform_vote define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_non_uniform_vote' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_ballot
#error "Missing cl_khr_subgroup_ballot"
#endif
#else
#ifdef cl_khr_subgroup_ballot
#error "Incorrect cl_khr_subgroup_ballot define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_ballot' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_non_uniform_arithmetic
#error "Missing cl_khr_subgroup_non_uniform_arithmetic"
#endif
#else
#ifdef cl_khr_subgroup_non_uniform_arithmetic
#error "Incorrect cl_khr_subgroup_non_uniform_arithmetic define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_non_uniform_arithmetic' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_arithmetic : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_shuffle
#error "Missing cl_khr_subgroup_shuffle"
#endif
#else
#ifdef cl_khr_subgroup_shuffle
#error "Incorrect cl_khr_subgroup_shuffle define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_shuffle' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_shuffle_relative
#error "Missing cl_khr_subgroup_shuffle_relative"
#endif
#else
#ifdef cl_khr_subgroup_shuffle_relative
#error "Incorrect cl_khr_subgroup_shuffle_relative define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_shuffle_relative' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable

#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#ifndef cl_khr_subgroup_clustered_reduce
#error "Missing cl_khr_subgroup_clustered_reduce"
#endif
#else
#ifdef cl_khr_subgroup_clustered_reduce
#error "Incorrect cl_khr_subgroup_clustered_reduce define"
#endif
// expected-warning@+2{{unsupported OpenCL extension 'cl_khr_subgroup_clustered_reduce' - ignoring}}
#endif
#pragma OPENCL EXTENSION cl_khr_subgroup_clustered_reduce : enable

