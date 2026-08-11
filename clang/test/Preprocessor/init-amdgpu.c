// Standalone amdgcn is LP64, so 'long' is used for the 64-bit and maximal
// integer types.
//
// RUN: %clang_cc1 -E -dM -triple=amdgpu9.0a-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s
// RUN: %clang_cc1 -x c++ -E -dM -triple=amdgpu9.42-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s
// RUN: %clang_cc1 -E -dM -triple=amdgpu10.30-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s
// RUN: %clang_cc1 -x c++ -E -dM -triple=amdgpu12.50-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s
//
// r600 has 32-bit pointers and keeps the generic defaults.
//
// RUN: %clang_cc1 -E -dM -triple=r600 < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LLP64 %s
//
// OpenCL mandates its own widths regardless of the target.
//
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -E -dM -triple=amdgpu9.0a-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=OPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -E -dM -triple=amdgpu12.50-amd-amdhsa < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=OPENCL %s
//
// Offloading languages copy these from the auxiliary host target, so the
// standalone definitions above do not leak into them.
//
// RUN: %clang_cc1 -x hip -fcuda-is-device -E -dM -triple=amdgpu9.0a-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -E -dM -triple=amdgpu9.0a-amd-amdhsa \
// RUN:   -aux-triple x86_64-pc-windows-msvc < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LLP64 %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -E -dM -triple=amdgpu12.50-amd-amdhsa \
// RUN:   -aux-triple i386-unknown-linux-gnu < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LLP64 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -E -dM \
// RUN:   -triple=amdgpu9.0a-amd-amdhsa -aux-triple x86_64-pc-windows-msvc < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LLP64 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -E -dM \
// RUN:   -triple=amdgpu12.50-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=LP64 %s

// LP64-DAG: #define __INT64_TYPE__ long int
// LP64-DAG: #define __UINT64_TYPE__ long unsigned int
// LP64-DAG: #define __INT64_C_SUFFIX__ L
// LP64-DAG: #define __INTMAX_TYPE__ long int
// LP64-DAG: #define __UINTMAX_TYPE__ long unsigned int
// LP64-DAG: #define __INTMAX_C_SUFFIX__ L
// LP64-DAG: #define __INTMAX_WIDTH__ 64
// LP64-DAG: #define __INTPTR_TYPE__ long int
// LP64-DAG: #define __SIZE_TYPE__ long unsigned int
// LP64-DAG: #define __PTRDIFF_TYPE__ long int

// LLP64-DAG: #define __INT64_TYPE__ long long int
// LLP64-DAG: #define __UINT64_TYPE__ long long unsigned int
// LLP64-DAG: #define __INT64_C_SUFFIX__ LL
// LLP64-DAG: #define __INTMAX_TYPE__ long long int
// LLP64-DAG: #define __UINTMAX_TYPE__ long long unsigned int
// LLP64-DAG: #define __INTMAX_C_SUFFIX__ LL
// LLP64-DAG: #define __INTMAX_WIDTH__ 64

// OPENCL-DAG: #define __INT64_TYPE__ long int
// OPENCL-DAG: #define __INTMAX_TYPE__ long long int
// OPENCL-DAG: #define __UINTMAX_TYPE__ long long unsigned int
// OPENCL-DAG: #define __INTPTR_TYPE__ long int
// OPENCL-DAG: #define __SIZE_TYPE__ long unsigned int
