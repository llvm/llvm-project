// Check that OpenCL image and sampler types are only available on AMDGPU
// targets with image instructions.

// No GPU specified (device libraries), GPUs with image instructions, and R600.
// RUN: %clang_cc1 -triple amdgpu-- -cl-std=CL2.0 -verify=img -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9.0a-- -cl-std=CL2.0 -verify=img -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu11.00-- -cl-std=CL3.0 -verify=img -fsyntax-only %s
// RUN: %clang_cc1 -triple r600-unknown-unknown -target-cpu cypress -cl-std=CL1.2 \
// RUN:   -verify=img -fsyntax-only %s

// GPUs without image instructions, before OpenCL C 3.0.
// RUN: %clang_cc1 -triple amdgpu9.42-- -cl-std=CL1.2 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9.42-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9.50-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu9.4-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.51-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.5-- -cl-std=CL2.0 -verify=noimg -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=clc++1.0 -verify=noimg -fsyntax-only %s

// GPUs without image instructions, OpenCL C 3.0 and C++ for OpenCL 2021.
// RUN: %clang_cc1 -triple amdgpu9.42-- -cl-std=CL3.0 -verify=nofeature -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=CL3.0 -verify=nofeature -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=clc++2021 -verify=nofeature -fsyntax-only %s

// Forcing the feature on does not make the types available.
// RUN: %clang_cc1 -triple amdgpu12.50-- -cl-std=CL3.0 -cl-ext=+__opencl_c_images \
// RUN:   -verify=noimg -fsyntax-only %s

// img-no-diagnostics

void test_ro(read_only image2d_t img);
// noimg-error@-1 {{'__read_only image2d_t' is not supported on this target}}
// nofeature-error@-2 {{use of type '__read_only image2d_t' requires __opencl_c_images support}}

void test_wo(write_only image2d_t img);
// noimg-error@-1 {{'__write_only image2d_t' is not supported on this target}}
// nofeature-error@-2 {{use of type '__write_only image2d_t' requires __opencl_c_images support}}

void test_sampler(sampler_t s);
// noimg-error@-1 {{'sampler_t' is not supported on this target}}
// nofeature-error@-2 {{use of type 'sampler_t' requires __opencl_c_images support}}

#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= 200)
void test_rw(read_write image2d_t img);
// noimg-error@-1 {{'__read_write image2d_t' is not supported on this target}}
// nofeature-error@-2 {{use of type '__read_write image2d_t' requires __opencl_c_images support}}

void test_3d_wo(write_only image3d_t img);
// noimg-error@-1 {{'__write_only image3d_t' is not supported on this target}}
// nofeature-error@-2 {{use of type '__write_only image3d_t' requires __opencl_c_images support}}
#endif
