// This test verifies that the correct macros are predefined.
//
// RUN: %clang_cc1 %s -E -dM -triple i686-pc-win32 -fms-compatibility \
// RUN:     -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS-STDINT
// CHECK-MS-STDINT:#define __INT16_MAX__ 32767
// CHECK-MS-STDINT:#define __INT32_MAX__ 2147483647
// CHECK-MS-STDINT:#define __INT64_MAX__ 9223372036854775807LL
// CHECK-MS-STDINT:#define __INT8_MAX__ 127
// CHECK-MS-STDINT:#define __INTPTR_MAX__ 2147483647
// CHECK-MS-STDINT:#define __INT_FAST16_MAX__ 32767
// CHECK-MS-STDINT:#define __INT_FAST16_TYPE__ short
// CHECK-MS-STDINT:#define __INT_FAST32_MAX__ 2147483647
// CHECK-MS-STDINT:#define __INT_FAST32_TYPE__ int
// CHECK-MS-STDINT:#define __INT_FAST64_MAX__ 9223372036854775807LL
// CHECK-MS-STDINT:#define __INT_FAST64_TYPE__ long long int
// CHECK-MS-STDINT:#define __INT_FAST8_MAX__ 127
// CHECK-MS-STDINT:#define __INT_FAST8_TYPE__ signed char
// CHECK-MS-STDINT:#define __INT_LEAST16_MAX__ 32767
// CHECK-MS-STDINT:#define __INT_LEAST16_TYPE__ short
// CHECK-MS-STDINT:#define __INT_LEAST32_MAX__ 2147483647
// CHECK-MS-STDINT:#define __INT_LEAST32_TYPE__ int
// CHECK-MS-STDINT:#define __INT_LEAST64_MAX__ 9223372036854775807LL
// CHECK-MS-STDINT:#define __INT_LEAST64_TYPE__ long long int
// CHECK-MS-STDINT:#define __INT_LEAST8_MAX__ 127
// CHECK-MS-STDINT:#define __INT_LEAST8_TYPE__ signed char
// CHECK-MS-STDINT-NOT:#define __UINT16_C_SUFFIX__ U
// CHECK-MS-STDINT:#define __UINT16_MAX__ 65535
// CHECK-MS-STDINT:#define __UINT16_TYPE__ unsigned short
// CHECK-MS-STDINT:#define __UINT32_C_SUFFIX__ U
// CHECK-MS-STDINT:#define __UINT32_MAX__ 4294967295U
// CHECK-MS-STDINT:#define __UINT32_TYPE__ unsigned int
// CHECK-MS-STDINT:#define __UINT64_C_SUFFIX__ ULL
// CHECK-MS-STDINT:#define __UINT64_MAX__ 18446744073709551615ULL
// CHECK-MS-STDINT:#define __UINT64_TYPE__ long long unsigned int
// CHECK-MS-STDINT-NOT:#define __UINT8_C_SUFFIX__ U
// CHECK-MS-STDINT:#define __UINT8_MAX__ 255
// CHECK-MS-STDINT:#define __UINT8_TYPE__ unsigned char
// CHECK-MS-STDINT:#define __UINTMAX_MAX__ 18446744073709551615ULL
// CHECK-MS-STDINT:#define __UINTPTR_MAX__ 4294967295U
// CHECK-MS-STDINT:#define __UINTPTR_TYPE__ unsigned int
// CHECK-MS-STDINT:#define __UINTPTR_WIDTH__ 32
// CHECK-MS-STDINT:#define __UINT_FAST16_MAX__ 65535
// CHECK-MS-STDINT:#define __UINT_FAST16_TYPE__ unsigned short
// CHECK-MS-STDINT:#define __UINT_FAST32_MAX__ 4294967295U
// CHECK-MS-STDINT:#define __UINT_FAST32_TYPE__ unsigned int
// CHECK-MS-STDINT:#define __UINT_FAST64_MAX__ 18446744073709551615ULL
// CHECK-MS-STDINT:#define __UINT_FAST64_TYPE__ long long unsigned int
// CHECK-MS-STDINT:#define __UINT_FAST8_MAX__ 255
// CHECK-MS-STDINT:#define __UINT_FAST8_TYPE__ unsigned char
// CHECK-MS-STDINT:#define __UINT_LEAST16_MAX__ 65535
// CHECK-MS-STDINT:#define __UINT_LEAST16_TYPE__ unsigned short
// CHECK-MS-STDINT:#define __UINT_LEAST32_MAX__ 4294967295U
// CHECK-MS-STDINT:#define __UINT_LEAST32_TYPE__ unsigned int
// CHECK-MS-STDINT:#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
// CHECK-MS-STDINT:#define __UINT_LEAST64_TYPE__ long long unsigned int
// CHECK-MS-STDINT:#define __UINT_LEAST8_MAX__ 255
// CHECK-MS-STDINT:#define __UINT_LEAST8_TYPE__ unsigned char
//
// RUN: %clang_cc1 %s -E -dM -ffast-math -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-FAST-MATH
// CHECK-FAST-MATH: #define __FAST_MATH__ 1
// CHECK-FAST-MATH: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang_cc1 %s -E -dM -ffinite-math-only -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-FINITE-MATH-ONLY
// CHECK-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang %s -E -dM -fno-finite-math-only -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-NO-FINITE-MATH-ONLY
// CHECK-NO-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 0
//
// RUN: %clang_cc1 %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-FINITE-MATH-FLAG-UNDEFINED
// CHECK-FINITE-MATH-FLAG-UNDEFINED: #define __FINITE_MATH_ONLY__ 0
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i386 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_I386
// CHECK-SYNC_CAS_I386-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i486 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_I486
// CHECK-SYNC_CAS_I486: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_I486: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_I486: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_I486-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
//
// RUN: %clang_cc1 %s -E -dM -o - -triple i686 -target-cpu i586 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_I586
// CHECK-SYNC_CAS_I586: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_I586: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_I586: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_I586: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
//
// RUN: %clang_cc1 %s -E -dM -o - -triple armv6 -target-cpu arm1136j-s \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_ARM
// CHECK-SYNC_CAS_ARM: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_ARM: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_ARM: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_ARM: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
//
// RUN: %clang_cc1 %s -E -dM -o - -triple armv7 -target-cpu cortex-a8 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_ARMv7
// CHECK-SYNC_CAS_ARMv7: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_ARMv7: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_ARMv7: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_ARMv7: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
//
// RUN: %clang_cc1 %s -E -dM -o - -triple armv6 -target-cpu cortex-m0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_ARMv6
// CHECK-SYNC_CAS_ARMv6-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP
//
// RUN: %clang_cc1 %s -E -dM -o - -triple mips -target-cpu mips2 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_MIPS \
// RUN:         --check-prefix=CHECK-SYNC_CAS_MIPS32
// RUN: %clang_cc1 %s -E -dM -o - -triple mips64 -target-cpu mips3 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_MIPS \
// RUN:         --check-prefix=CHECK-SYNC_CAS_MIPS64
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_MIPS32-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
// CHECK-SYNC_CAS_MIPS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL10
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL1.1 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL11
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL1.2 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL12
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL2.0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL20
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-fast-relaxed-math \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-FRM
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=c++ \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CLCPP10
// CHECK-CL10: #define CL_VERSION_1_0 100
// CHECK-CL10: #define CL_VERSION_1_1 110
// CHECK-CL10: #define CL_VERSION_1_2 120
// CHECK-CL10: #define CL_VERSION_2_0 200
// CHECK-CL10: #define __OPENCL_C_VERSION__ 100
// CHECK-CL10-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL11: #define CL_VERSION_1_0 100
// CHECK-CL11: #define CL_VERSION_1_1 110
// CHECK-CL11: #define CL_VERSION_1_2 120
// CHECK-CL11: #define CL_VERSION_2_0 200
// CHECK-CL11: #define __OPENCL_C_VERSION__ 110
// CHECK-CL11-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL12: #define CL_VERSION_1_0 100
// CHECK-CL12: #define CL_VERSION_1_1 110
// CHECK-CL12: #define CL_VERSION_1_2 120
// CHECK-CL12: #define CL_VERSION_2_0 200
// CHECK-CL12: #define __OPENCL_C_VERSION__ 120
// CHECK-CL12-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL20: #define CL_VERSION_1_0 100
// CHECK-CL20: #define CL_VERSION_1_1 110
// CHECK-CL20: #define CL_VERSION_1_2 120
// CHECK-CL20: #define CL_VERSION_2_0 200
// CHECK-CL20: #define __OPENCL_C_VERSION__ 200
// CHECK-CL20-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-FRM: #define __FAST_RELAXED_MATH__ 1
// CHECK-CLCPP10: #define __CL_CPP_VERSION_1_0__ 100
// CHECK-CLCPP10: #define __OPENCL_CPP_VERSION__ 100
// CHECK-CLCPP10-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CLCPP10-NOT: #define __ENDIAN_LITTLE__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl \
// RUN:   | FileCheck %s --check-prefix=MSCOPE
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_DEVICE 2
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spir-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIR
// CHECK-SPIR: #define __IMAGE_SUPPORT__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP
// CHECK-HIP-NOT: #define __CUDA_ARCH__
// CHECK-HIP: #define __HIPCC__ 1
// CHECK-HIP-NOT: #define __HIP_DEVICE_COMPILE__ 1
// CHECK-HIP: #define __HIP__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP-DEV
// CHECK-HIP-DEV-NOT: #define __CUDA_ARCH__
// CHECK-HIP-DEV: #define __HIPCC__ 1
// CHECK-HIP-DEV: #define __HIP_DEVICE_COMPILE__ 1
// CHECK-HIP-DEV: #define __HIP__ 1
