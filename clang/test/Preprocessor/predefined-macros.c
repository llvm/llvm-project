// This test verifies that the correct macros are predefined.
//
// RUN: %clang_cc1 %s -x c++ -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++1z -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS
// CHECK-MS: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS: #define _MSC_EXTENSIONS 1
// CHECK-MS: #define _MSC_VER 1900
// CHECK-MS: #define _MSVC_LANG 201403L
// CHECK-MS: #define _M_IX86 600
// CHECK-MS: #define _M_IX86_FP 0
// CHECK-MS: #define _WIN32 1
// CHECK-MS-NOT: #define __STRICT_ANSI__
// CHECK-MS-NOT: GCC
// CHECK-MS-NOT: GNU
// CHECK-MS-NOT: GXX
//
// RUN: %clang_cc1 %s -x c++ -E -dM -triple x86_64-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fms-compatibility-version=19.00 -std=c++14 -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// CHECK-MS64: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS64: #define _MSC_EXTENSIONS 1
// CHECK-MS64: #define _MSC_VER 1900
// CHECK-MS64: #define _MSVC_LANG 201402L
// CHECK-MS64: #define _M_AMD64 100
// CHECK-MS64: #define _M_X64 100
// CHECK-MS64: #define _WIN64 1
// CHECK-MS64-NOT: #define __STRICT_ANSI__
// CHECK-MS64-NOT: GCC
// CHECK-MS64-NOT: GNU
// CHECK-MS64-NOT: GXX
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

// RUN: %clang_cc1 %s -E -dM -o - -x cl \
// RUN:   | FileCheck %s --check-prefix=MSCOPE
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_DEVICE 2
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
// MSCOPE:#define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0

// RUN: %clang_cc1 -triple i386-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-X86-WIN

// CHECK-X86-WIN-NOT: #define WIN32 1
// CHECK-X86-WIN-NOT: #define WIN64 1
// CHECK-X86-WIN-NOT: #define WINNT 1
// CHECK-X86-WIN: #define _WIN32 1
// CHECK-X86-WIN-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple thumbv7-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM-WIN

// CHECK-ARM-WIN-NOT: #define WIN32 1
// CHECK-ARM-WIN-NOT: #define WIN64 1
// CHECK-ARM-WIN-NOT: #define WINNT 1
// CHECK-ARM-WIN: #define _WIN32 1
// CHECK-ARM-WIN-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple x86_64-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-AMD64-WIN

// CHECK-AMD64-WIN-NOT: #define WIN32 1
// CHECK-AMD64-WIN-NOT: #define WIN64 1
// CHECK-AMD64-WIN-NOT: #define WINNT 1
// CHECK-AMD64-WIN: #define _WIN32 1
// CHECK-AMD64-WIN: #define _WIN64 1

// RUN: %clang_cc1 -triple aarch64-windows %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64-WIN

// CHECK-ARM64-WIN-NOT: #define WIN32 1
// CHECK-ARM64-WIN-NOT: #define WIN64 1
// CHECK-ARM64-WIN-NOT: #define WINNT 1
// CHECK-ARM64-WIN: #define _M_ARM64 1
// CHECK-ARM64-WIN: #define _WIN32 1
// CHECK-ARM64-WIN: #define _WIN64 1

// RUN: %clang_cc1 -triple i686-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-X86-MINGW

// CHECK-X86-MINGW: #define WIN32 1
// CHECK-X86-MINGW-NOT: #define WIN64 1
// CHECK-X86-MINGW: #define WINNT 1
// CHECK-X86-MINGW: #define _WIN32 1
// CHECK-X86-MINGW-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple thumbv7-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM-MINGW

// CHECK-ARM-MINGW: #define WIN32 1
// CHECK-ARM-MINGW-NOT: #define WIN64 1
// CHECK-ARM-MINGW: #define WINNT 1
// CHECK-ARM-MINGW: #define _WIN32 1
// CHECK-ARM-MINGW-NOT: #define _WIN64 1

// RUN: %clang_cc1 -triple x86_64-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-AMD64-MINGW

// CHECK-AMD64-MINGW: #define WIN32 1
// CHECK-AMD64-MINGW: #define WIN64 1
// CHECK-AMD64-MINGW: #define WINNT 1
// CHECK-AMD64-MINGW: #define _WIN32 1
// CHECK-AMD64-MINGW: #define _WIN64 1

// RUN: %clang_cc1 -triple aarch64-windows-gnu %s -E -dM -o - \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-ARM64-MINGW

// CHECK-ARM64-MINGW-NOT: #define _M_ARM64 1
// CHECK-ARM64-MINGW: #define WIN32 1
// CHECK-ARM64-MINGW: #define WIN64 1
// CHECK-ARM64-MINGW: #define WINNT 1
// CHECK-ARM64-MINGW: #define _WIN32 1
// CHECK-ARM64-MINGW: #define _WIN64 1
// CHECK-ARM64-MINGW: #define __aarch64__ 1

// RUN: %clang_cc1 %s -E -dM -o - -x cl -triple spir-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=CHECK-SPIR
// CHECK-SPIR: #define __IMAGE_SUPPORT__ 1
