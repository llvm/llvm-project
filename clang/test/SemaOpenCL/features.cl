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

// For OpenCL C 2.0 feature macros, test that CL2.0 define them but earlier OpenCL
// versions don't define feature macros accidentally.
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.1 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL1.2 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=NO-FEATURES
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL2.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES-CL20
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=clc++1.0 \
// RUN:   | FileCheck -match-full-lines %s  --check-prefix=FEATURES-CL20

// Features can be disabled using -cl-ext=-<feature>.
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header \
// RUN:    -cl-ext=-__opencl_c_integer_dot_product_input_4x8bit \
// RUN:    -cl-ext=-__opencl_c_integer_dot_product_input_4x8bit_packed \
// RUN:   | FileCheck %s --check-prefix=NO-FEATURES-CL20
// RUN: %clang_cc1 -triple spir-unknown-unknown %s -E -dM -o - -x cl -cl-std=CL3.0 -fdeclare-opencl-builtins -finclude-default-header \
// RUN:    -cl-ext=-__opencl_c_work_group_collective_functions, \
// RUN:    -cl-ext=-__opencl_c_atomic_order_seq_cst \
// RUN:    -cl-ext=-__opencl_c_atomic_scope_device \
// RUN:    -cl-ext=-__opencl_c_atomic_scope_all_devices \
// RUN:    -cl-ext=-__opencl_c_read_write_images \
// RUN:   | FileCheck %s --check-prefix=DISABLE-FEATURES

// Note that __opencl_c_int64 is always defined assuming
// always compiling for FULL OpenCL profile

// FEATURES: #define __opencl_c_3d_image_writes 1
// FEATURES: #define __opencl_c_atomic_order_acq_rel 1
// FEATURES: #define __opencl_c_atomic_order_seq_cst 1
// FEATURES: #define __opencl_c_atomic_scope_all_devices 1
// FEATURES: #define __opencl_c_atomic_scope_device 1
// FEATURES: #define __opencl_c_device_enqueue 1
// FEATURES: #define __opencl_c_ext_image_unorm_int_2_101010 1
// FEATURES: #define __opencl_c_ext_image_unsigned_10x6_12x4_14x2 1
// FEATURES: #define __opencl_c_fp64 1
// FEATURES: #define __opencl_c_generic_address_space 1
// FEATURES: #define __opencl_c_images 1
// FEATURES: #define __opencl_c_int64 1
// FEATURES: #define __opencl_c_integer_dot_product_input_4x8bit 1
// FEATURES: #define __opencl_c_integer_dot_product_input_4x8bit_packed 1
// FEATURES: #define __opencl_c_kernel_clock_scope_device 1
// FEATURES: #define __opencl_c_kernel_clock_scope_sub_group 1
// FEATURES: #define __opencl_c_kernel_clock_scope_work_group 1
// FEATURES: #define __opencl_c_pipes 1
// FEATURES: #define __opencl_c_program_scope_global_variables 1
// FEATURES: #define __opencl_c_read_write_images 1
// FEATURES: #define __opencl_c_subgroups 1
// FEATURES: #define __opencl_c_work_group_collective_functions 1

// FEATURES-CL20: #define __opencl_c_atomic_order_acq_rel 1
// FEATURES-CL20: #define __opencl_c_atomic_order_seq_cst 1
// FEATURES-CL20: #define __opencl_c_atomic_scope_all_devices 1
// FEATURES-CL20: #define __opencl_c_atomic_scope_device 1
// FEATURES-CL20: #define __opencl_c_device_enqueue 1
// FEATURES-CL20: #define __opencl_c_generic_address_space 1
// FEATURES-CL20: #define __opencl_c_images 1
// FEATURES-CL20: #define __opencl_c_int64 1
// FEATURES-CL20: #define __opencl_c_pipes 1
// FEATURES-CL20: #define __opencl_c_program_scope_global_variables 1
// FEATURES-CL20: #define __opencl_c_read_write_images 1
// FEATURES-CL20: #define __opencl_c_work_group_collective_functions 1

// NO-FEATURES: #define __opencl_c_int64 1
// NO-FEATURES-NOT: #define __opencl_c_3d_image_writes 1
// NO-FEATURES-NOT: #define __opencl_c_atomic_order_acq_rel 1
// NO-FEATURES-NOT: #define __opencl_c_atomic_order_seq_cst 1
// NO-FEATURES-NOT: #define __opencl_c_atomic_scope_all_devices 1
// NO-FEATURES-NOT: #define __opencl_c_atomic_scope_device 1
// NO-FEATURES-NOT: #define __opencl_c_device_enqueue 1
// NO-FEATURES-NOT: #define __opencl_c_ext_image_unorm_int_2_101010 1
// NO-FEATURES-NOT: #define __opencl_c_ext_image_unsigned_10x6_12x4_14x2 1
// NO-FEATURES-NOT: #define __opencl_c_fp64 1
// NO-FEATURES-NOT: #define __opencl_c_generic_address_space 1
// NO-FEATURES-NOT: #define __opencl_c_images 1
// NO-FEATURES-NOT: #define __opencl_c_int64 1
// NO-FEATURES-NOT: #define __opencl_c_integer_dot_product_input_4x8bit 1
// NO-FEATURES-NOT: #define __opencl_c_integer_dot_product_input_4x8bit_packed 1
// NO-FEATURES-NOT: #define __opencl_c_kernel_clock_scope_device 1
// NO-FEATURES-NOT: #define __opencl_c_kernel_clock_scope_sub_group 1
// NO-FEATURES-NOT: #define __opencl_c_kernel_clock_scope_work_group 1
// NO-FEATURES-NOT: #define __opencl_c_pipes 1
// NO-FEATURES-NOT: #define __opencl_c_program_scope_global_variables 1
// NO-FEATURES-NOT: #define __opencl_c_read_write_images 1
// NO-FEATURES-NOT: #define __opencl_c_subgroups 1
// NO-FEATURES-NOT: #define __opencl_c_work_group_collective_functions 1

// NO-FEATURES-CL20-NOT: #define __opencl_c_integer_dot_product_input_4x8bit
// NO-FEATURES-CL20-NOT: #define __opencl_c_integer_dot_product_input_4x8bit_packed

// DISABLE-FEATURES-NOT: #define __opencl_c_work_group_collective_functions
// DISABLE-FEATURES-NOT: #define __opencl_c_atomic_order_seq_cst
// DISABLE-FEATURES-NOT: #define __opencl_c_atomic_scope_all_devices
// DISABLE-FEATURES-NOT: #define __opencl_c_atomic_scope_device
// DISABLE-FEATURES-NOT: #define __opencl_c_read_write_images
