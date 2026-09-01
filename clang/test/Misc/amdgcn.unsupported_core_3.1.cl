// RUN: %clang_cc1 -cl-std=CL3.1 -triple amdgcn-amd-amdhsa -Wpedantic-core-features %s 2>&1 | FileCheck %s

// CHECK: cl_khr_extended_bit_ops is a core feature in OpenCL C version 3.1 but not supported on this target
// CHECK: cl_khr_integer_dot_product is a core feature in OpenCL C version 3.1 but not supported on this target
// CHECK: cl_khr_subgroup_extended_types is a core feature in OpenCL C version 3.1 but not supported on this target
// CHECK: cl_khr_subgroup_rotate is a core feature in OpenCL C version 3.1 but not supported on this target
// CHECK: cl_khr_subgroup_shuffle_relative is a core feature in OpenCL C version 3.1 but not supported on this target
// CHECK: cl_khr_subgroup_shuffle is a core feature in OpenCL C version 3.1 but not supported on this target
