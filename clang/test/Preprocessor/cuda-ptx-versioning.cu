// RUN: %clang_cc1 %s -E -dM -o - -x cuda -fcuda-is-device -triple nvptx64 \
// RUN: | FileCheck -match-full-lines %s --check-prefix=CHECK-CUDA32
// CHECK-CUDA32: #define __PTX_VERSION__ 32

// RUN: %clang_cc1 %s -E -dM -o - -x cuda -fcuda-is-device -triple nvptx64 -target-feature +ptx78 \
// RUN:  -target-cpu sm_90 | FileCheck -match-full-lines %s --check-prefix=CHECK-CUDA78
// CHECK-CUDA78: #define __PTX_VERSION__ 78

// RUN: %clang_cc1 %s -E -dM -o - -x cuda -fcuda-is-device -triple nvptx64 -target-feature +ptx80 \
// RUN:  -target-cpu sm_80 | FileCheck -match-full-lines %s --check-prefix=CHECK-CUDA80
// CHECK-CUDA80: #define __PTX_VERSION__ 80
