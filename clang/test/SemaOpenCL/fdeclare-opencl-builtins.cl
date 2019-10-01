// RUN: %clang_cc1 %s -triple spir -verify -pedantic -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins

// Test the -fdeclare-opencl-builtins option.

typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef unsigned int uint;
typedef __SIZE_TYPE__ size_t;

kernel void basic_conversion(global float4 *buf, global int4 *res) {
  res[0] = convert_int4(buf[0]);
}

kernel void basic_readonly_image_type(__read_only image2d_t img, int2 coord, global float4 *out) {
  out[0] = read_imagef(img, coord);
}

kernel void basic_subgroup(global uint *out) {
  out[0] = get_sub_group_size();
// expected-error@-1{{use of declaration 'get_sub_group_size' requires cl_khr_subgroups extension to be enabled}}
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
  out[1] = get_sub_group_size();
}
