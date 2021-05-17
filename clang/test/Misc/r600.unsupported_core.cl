// RUN: %clang_cc1 -cl-std=CL2.0 -triple r600-unknown-unknown -Wpedantic-core-features %s 2> %t
// RUN: FileCheck < %t %s

// CHECK: cl_khr_byte_addressable_store is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK: cl_khr_global_int32_base_atomics is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK: cl_khr_global_int32_extended_atomics is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK: cl_khr_local_int32_base_atomics is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK: cl_khr_local_int32_extended_atomics is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK: cl_khr_3d_image_writes is a core feature in OpenCL C version 2.0 but not supported on this target
