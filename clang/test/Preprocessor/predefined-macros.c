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
// CHECK-FAST-MATH: #define __NO_MATH_ERRNO__ 1
//
// RUN: %clang_cc1 %s -E -dM -fmath-errno -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-MATH-ERRNO
// CHECK-MATH-ERRNO-NOT: __NO_MATH_ERRNO__
//
// RUN: %clang %s -E -dM -fno-math-errno -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-NO-MATH-ERRNO
// CHECK-NO-MATH-ERRNO: #define __NO_MATH_ERRNO__ 1
//
// RUN: %clang_cc1 %s -E -dM -menable-no-nans -menable-no-infs -o - \
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
// RUN: %clang_cc1 %s -E -dM -o - -triple mips -target-cpu mips1 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_MIPS32_MIPS1
// RUN: %clang_cc1 %s -E -dM -o - -triple mips -target-cpu mips2 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_MIPS \
// RUN:         --check-prefix=CHECK-SYNC_CAS_MIPS32
// RUN: %clang_cc1 %s -E -dM -o - -triple mips64 -target-cpu mips3 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SYNC_CAS_MIPS \
// RUN:         --check-prefix=CHECK-SYNC_CAS_MIPS64
// CHECK-SYNC_CAS_MIPS32_MIPS1-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK-SYNC_CAS_MIPS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-SYNC_CAS_MIPS32-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
// CHECK-SYNC_CAS_MIPS64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL12
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL1.0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL10
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL1.1 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL11
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL1.2 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL12
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL2.0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL20
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=CL3.0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CL30
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-fast-relaxed-math \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-FRM
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=clc++ \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CLCPP10
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=clc++1.0 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CLCPP10
// RUN: %clang_cc1 %s -E -dM -o - -x cl -cl-std=clc++2021 \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CLCPP2021
// CHECK-CL10: #define CL_VERSION_1_0 100
// CHECK-CL10: #define CL_VERSION_1_1 110
// CHECK-CL10: #define CL_VERSION_1_2 120
// CHECK-CL10: #define CL_VERSION_2_0 200
// CHECK-CL10: #define CL_VERSION_3_0 300
// CHECK-CL10: #define __OPENCL_C_VERSION__ 100
// CHECK-CL10-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL11: #define CL_VERSION_1_0 100
// CHECK-CL11: #define CL_VERSION_1_1 110
// CHECK-CL11: #define CL_VERSION_1_2 120
// CHECK-CL11: #define CL_VERSION_2_0 200
// CHECK-CL11: #define CL_VERSION_3_0 300
// CHECK-CL11: #define __OPENCL_C_VERSION__ 110
// CHECK-CL11-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL12: #define CL_VERSION_1_0 100
// CHECK-CL12: #define CL_VERSION_1_1 110
// CHECK-CL12: #define CL_VERSION_1_2 120
// CHECK-CL12: #define CL_VERSION_2_0 200
// CHECK-CL12: #define CL_VERSION_3_0 300
// CHECK-CL12: #define __OPENCL_C_VERSION__ 120
// CHECK-CL12-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL20: #define CL_VERSION_1_0 100
// CHECK-CL20: #define CL_VERSION_1_1 110
// CHECK-CL20: #define CL_VERSION_1_2 120
// CHECK-CL20: #define CL_VERSION_2_0 200
// CHECK-CL20: #define CL_VERSION_3_0 300
// CHECK-CL20: #define __OPENCL_C_VERSION__ 200
// CHECK-CL20-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CL30: #define CL_VERSION_1_0 100
// CHECK-CL30: #define CL_VERSION_1_1 110
// CHECK-CL30: #define CL_VERSION_1_2 120
// CHECK-CL30: #define CL_VERSION_2_0 200
// CHECK-CL30: #define CL_VERSION_3_0 300
// CHECK-CL30: #define __OPENCL_C_VERSION__ 300
// CHECK-CL30-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-FRM: #define __FAST_RELAXED_MATH__ 1
// CHECK-CLCPP10: #define __CL_CPP_VERSION_1_0__ 100
// CHECK-CLCPP10: #define __CL_CPP_VERSION_2021__ 202100
// CHECK-CLCPP10: #define __OPENCL_CPP_VERSION__ 100
// CHECK-CLCPP10-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CLCPP10-NOT: #define __ENDIAN_LITTLE__ 1
// CHECK-CLCPP2021: #define __CL_CPP_VERSION_1_0__ 100
// CHECK-CLCPP2021: #define __CL_CPP_VERSION_2021__ 202100
// CHECK-CLCPP2021: #define __OPENCL_CPP_VERSION__ 202100
// CHECK-CLCPP2021-NOT: #define __FAST_RELAXED_MATH__ 1
// CHECK-CLCPP2021-NOT: #define __ENDIAN_LITTLE__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl \
// RUN:   | FileCheck %s --check-prefix=MSCOPE
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_DEVICE 2
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spir-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIR
// CHECK-SPIR-DAG: #define __IMAGE_SUPPORT__ 1
// CHECK-SPIR-DAG: #define __SPIR__ 1
// CHECK-SPIR-DAG: #define __SPIR32__ 1
// CHECK-SPIR-NOT: #define __SPIR64__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spir64-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIR64
// CHECK-SPIR64-DAG: #define __IMAGE_SUPPORT__ 1
// CHECK-SPIR64-DAG: #define __SPIR__ 1
// CHECK-SPIR64-DAG: #define __SPIR64__ 1
// CHECK-SPIR64-NOT: #define __SPIR32__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spirv32-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIRV32
// CHECK-SPIRV32-DAG: #define __IMAGE_SUPPORT__ 1
// CHECK-SPIRV32-DAG: #define __SPIRV__ 1
// CHECK-SPIRV32-DAG: #define __SPIRV32__ 1
// CHECK-SPIRV32-NOT: #define __SPIRV64__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spirv64-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIRV64
// CHECK-SPIRV64-DAG: #define __IMAGE_SUPPORT__ 1
// CHECK-SPIRV64-DAG: #define __SPIRV__ 1
// CHECK-SPIRV64-DAG: #define __SPIRV64__ 1
// CHECK-SPIRV64-NOT: #define __SPIRV32__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spirv64-amd-amdhsa \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIRV64-AMDGCN
// CHECK-SPIRV64-AMDGCN-DAG: #define __IMAGE_SUPPORT__ 1
// CHECK-SPIRV64-AMDGCN-DAG: #define __SPIRV__ 1
// CHECK-SPIRV64-AMDGCN-DAG: #define __SPIRV64__ 1
// CHECK-SPIRV64-AMDGCN-DAG: #define __AMD__ 1
// CHECK-SPIRV64-AMDGCN-DAG: #define __AMDGCN__ 1
// CHECK-SPIRV64-AMDGCN-DAG: #define __AMDGPU__ 1
// CHECK-SPIRV64-AMDGCN-NOT: #define __SPIRV32__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP
// CHECK-HIP: #define __HIPCC__ 1
// CHECK-HIP: #define __HIP__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cuda -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CUDA-NEG
// CHECK-CUDA-NEG-NOT: #define __CLANG_RDC__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP-NEG
// CHECK-HIP-NEG-NOT: #define __CUDA_ARCH__
// CHECK-HIP-NEG-NOT: #define __HIP_DEVICE_COMPILE__ 1
// CHECK-HIP-NEG-NOT: #define __CLANG_RDC__ 1
// CHECK-HIP-NEG-NOT: #define HIP_API_PER_THREAD_DEFAULT_STREAM

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP-DEV
// CHECK-HIP-DEV: #define __HIPCC__ 1
// CHECK-HIP-DEV: #define __HIP_DEVICE_COMPILE__ 1
// CHECK-HIP-DEV: #define __HIP__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cuda -triple nvptx \
// RUN:   -fcuda-is-device \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-CUDA-DEV-NEG
// CHECK-CUDA-DEV-NEG-NOT: #define __CLANG_RDC__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIP-DEV-NEG
// CHECK-HIP-DEV-NEG-NOT: #define __CUDA_ARCH__
// CHECK-HIP-DEV-NEG-NOT: #define __CLANG_RDC__ 1
// CHECK-HIP-DEV-NEG-NOT: #define HIP_API_PER_THREAD_DEFAULT_STREAM

// RUN: %clang_cc1 %s -E -dM -o - -x cuda -triple x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc | FileCheck %s --check-prefix=CHECK-RDC
// RUN: %clang_cc1 %s -E -dM -o - -x cuda -triple nvptx \
// RUN:   -fgpu-rdc -fcuda-is-device \
// RUN:   | FileCheck %s --check-prefix=CHECK-RDC
// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc | FileCheck %s --check-prefix=CHECK-RDC
// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -fgpu-rdc -fcuda-is-device \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-RDC
// CHECK-RDC: #define __CLANG_RDC__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   -fgpu-default-stream=per-thread \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-PTH
// RUN: %clang_cc1 %s -E -dM -o - -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -fgpu-default-stream=per-thread \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-PTH
// CHECK-PTH: #define HIP_API_PER_THREAD_DEFAULT_STREAM 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip --hipstdpar -triple x86_64-unknown-linux-gnu \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-HIPSTDPAR
// CHECK-HIPSTDPAR: #define __HIPSTDPAR__ 1
// CHECK-HIPSTDPAR-NOT: #define __HIPSTDPAR_INTERPOSE_ALLOC__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip --hipstdpar --hipstdpar-interpose-alloc \
// RUN:  -triple x86_64-unknown-linux-gnu | FileCheck -match-full-lines %s \
// RUN:  --check-prefix=CHECK-HIPSTDPAR-INTERPOSE
// CHECK-HIPSTDPAR-INTERPOSE: #define __HIPSTDPAR_INTERPOSE_ALLOC__ 1
// CHECK-HIPSTDPAR-INTERPOSE: #define __HIPSTDPAR__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x hip --hipstdpar --hipstdpar-interpose-alloc \
// RUN:  -triple amdgcn-amd-amdhsa -fcuda-is-device | FileCheck -match-full-lines \
// RUN:  %s --check-prefix=CHECK-HIPSTDPAR-INTERPOSE-DEV-NEG
// CHECK-HIPSTDPAR-INTERPOSE-DEV-NEG: #define __HIPSTDPAR__ 1
// CHECK-HIPSTDPAR-INTERPOSE-DEV-NEG-NOT: #define __HIPSTDPAR_INTERPOSE_ALLOC__ 1
