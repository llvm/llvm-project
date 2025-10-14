// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=-all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++2021 -cl-ext=-all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++2021 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++2021 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple r600-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++2021 -cl-ext=+all \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES

// For OpenCL C 2.0 feature macros are defined only in header, so test that earlier OpenCL
// versions don't define feature macros accidentally and CL2.0 don't define them without header
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.1 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.2 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL2.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++1.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES

// For OpenCL C 2.0, header-only features can be disabled using macros.
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header \
// RUN:    -D__undef___opencl_c_integer_dot_product_input_4x8bit \
// RUN:    -D__undef___opencl_c_integer_dot_product_input_4x8bit_packed \
// RUN:   | FileCheck %s --check-prefix=NO-HEADERONLY-FEATURES-CL20

// For OpenCL C 3.0, header-only features can be disabled using macros.
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -fdeclare-opencl-builtins -finclude-default-header \
// RUN:    -D__undef___opencl_c_work_group_collective_functions=1 \
// RUN:    -D__undef___opencl_c_atomic_order_seq_cst=1 \
// RUN:    -D__undef___opencl_c_atomic_scope_device=1 \
// RUN:    -D__undef___opencl_c_atomic_scope_all_devices=1 \
// RUN:    -D__undef___opencl_c_read_write_images=1 \
// RUN:   | FileCheck %s --check-prefix=NO-HEADERONLY-FEATURES

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
// NO-FEATURES-NOT: #define __opencl_c_3d_image_writes
// NO-FEATURES-NOT: #define __opencl_c_atomic_order_acq_rel
// NO-FEATURES-NOT: #define __opencl_c_atomic_order_seq_cst
// NO-FEATURES-NOT: #define __opencl_c_device_enqueue
// NO-FEATURES-NOT: #define __opencl_c_fp64
// NO-FEATURES-NOT: #define __opencl_c_generic_address_space
// NO-FEATURES-NOT: #define __opencl_c_images
// NO-FEATURES-NOT: #define __opencl_c_pipes
// NO-FEATURES-NOT: #define __opencl_c_program_scope_global_variables
// NO-FEATURES-NOT: #define __opencl_c_read_write_images
// NO-FEATURES-NOT: #define __opencl_c_subgroups

// NO-HEADERONLY-FEATURES-CL20-NOT: #define __opencl_c_integer_dot_product_input_4x8bit
// NO-HEADERONLY-FEATURES-CL20-NOT: #define __opencl_c_integer_dot_product_input_4x8bit_packed

// NO-HEADERONLY-FEATURES-NOT: #define __opencl_c_work_group_collective_functions
// NO-HEADERONLY-FEATURES-NOT: #define __opencl_c_atomic_order_seq_cst
// NO-HEADERONLY-FEATURES-NOT: #define __opencl_c_atomic_scope_device
// NO-HEADERONLY-FEATURES-NOT: #define __opencl_c_atomic_scope_all_devices
// NO-HEADERONLY-FEATURES-NOT: #define __opencl_c_read_write_images
