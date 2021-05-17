// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=-all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES

// For OpenCL C 2.0 feature macros are defined only in header, so test that earlier OpenCL
// versions don't define feature macros accidentally and CL2.0 don't define them without header
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.1 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.2 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL2.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CLC++ \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES

// Note that __opencl_c_int64 is always defined assuming
// always compiling for FULL OpenCL profile

// FEATURES: #define __opencl_c_3d_image_writes 1
// FEATURES: #define __opencl_c_atomic_order_acq_rel 1
// FEATURES: #define __opencl_c_atomic_order_seq_cst 1
// FEATURES: #define __opencl_c_device_enqueue 1
// FEATURES: #define __opencl_c_fp64 1
// FEATURES: #define __opencl_c_generic_address_space 1
// FEATURES: #define __opencl_c_images 1
// FEATURES: #define __opencl_c_int64 1
// FEATURES: #define __opencl_c_pipes 1
// FEATURES: #define __opencl_c_program_scope_global_variables 1
// FEATURES: #define __opencl_c_read_write_images 1
// FEATURES: #define __opencl_c_subgroups 1

// NO-FEATURES: #define __opencl_c_int64 1
// NO-FEATURES-NOT: __opencl_c_3d_image_writes
// NO-FEATURES-NOT: __opencl_c_atomic_order_acq_rel
// NO-FEATURES-NOT: __opencl_c_atomic_order_seq_cst
// NO-FEATURES-NOT: __opencl_c_device_enqueue
// NO-FEATURES-NOT: __opencl_c_fp64
// NO-FEATURES-NOT: __opencl_c_generic_address_space
// NO-FEATURES-NOT: __opencl_c_images
// NO-FEATURES-NOT: __opencl_c_pipes
// NO-FEATURES-NOT: __opencl_c_program_scope_global_variables
// NO-FEATURES-NOT: __opencl_c_read_write_images
// NO-FEATURES-NOT: __opencl_c_subgroups
