/// Tests legacy SPIR target adaptation of type properties from the host.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=SPIR64-LINUX %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=SPIR64-WIN %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -aux-triple i386-unknown-linux-gnu \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=SPIR32-LINUX %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=SPIR64-NOHOST %s

// SPIR64 + Linux (LP64)
// SPIR64-LINUX-DAG: #define __SIZE_TYPE__ long unsigned int
// SPIR64-LINUX-DAG: #define __PTRDIFF_TYPE__ long int
// SPIR64-LINUX-DAG: #define __INTPTR_TYPE__ long int
// SPIR64-LINUX-DAG: #define __SIZEOF_LONG__ 8
// SPIR64-LINUX-DAG: #define __SIZEOF_POINTER__ 8

// SPIR64 + Windows (LLP64)
// SPIR64-WIN-DAG: #define __SIZE_TYPE__ long long unsigned int
// SPIR64-WIN-DAG: #define __PTRDIFF_TYPE__ long long int
// SPIR64-WIN-DAG: #define __INTPTR_TYPE__ long long int
// SPIR64-WIN-DAG: #define __SIZEOF_LONG__ 4
// SPIR64-WIN-DAG: #define __SIZEOF_POINTER__ 8

// SPIR32 + Linux i386 (ILP32)
// SPIR32-LINUX-DAG: #define __SIZE_TYPE__ unsigned int
// SPIR32-LINUX-DAG: #define __PTRDIFF_TYPE__ int
// SPIR32-LINUX-DAG: #define __INTPTR_TYPE__ int
// SPIR32-LINUX-DAG: #define __SIZEOF_POINTER__ 4

// SPIR64 no host (defaults)
// SPIR64-NOHOST-DAG: #define __SIZE_TYPE__ long unsigned int
// SPIR64-NOHOST-DAG: #define __PTRDIFF_TYPE__ long int
// SPIR64-NOHOST-DAG: #define __INTPTR_TYPE__ long int
// SPIR64-NOHOST-DAG: #define __SIZEOF_POINTER__ 8
