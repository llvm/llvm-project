/// Tests SPIR-V device target adaptation of SizeType, PtrDiffType, and
/// IntPtrType from the host target via -aux-triple.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=LINUX64 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=WIN64 %s
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -aux-triple i386-unknown-linux-gnu \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=LINUX32 %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=NOHOST64 %s
// RUN: %clang_cc1 -triple spirv32-unknown-unknown \
// RUN:   -fsycl-is-device -E -dM %s | FileCheck --check-prefix=NOHOST32 %s

// Linux x86_64 host (LP64)
// LINUX64-DAG: #define __SIZE_TYPE__ long unsigned int
// LINUX64-DAG: #define __PTRDIFF_TYPE__ long int
// LINUX64-DAG: #define __INTPTR_TYPE__ long int
// LINUX64-DAG: #define __SIZEOF_SIZE_T__ 8
// LINUX64-DAG: #define __SIZEOF_PTRDIFF_T__ 8
// LINUX64-DAG: #define __SIZEOF_LONG__ 8
// LINUX64-DAG: #define __SIZEOF_POINTER__ 8

// Windows x86_64 host (LLP64)
// WIN64-DAG: #define __SIZE_TYPE__ long long unsigned int
// WIN64-DAG: #define __PTRDIFF_TYPE__ long long int
// WIN64-DAG: #define __INTPTR_TYPE__ long long int
// WIN64-DAG: #define __SIZEOF_SIZE_T__ 8
// WIN64-DAG: #define __SIZEOF_PTRDIFF_T__ 8
// WIN64-DAG: #define __SIZEOF_LONG__ 4
// WIN64-DAG: #define __SIZEOF_POINTER__ 8

// Linux i386 host (ILP32)
// LINUX32-DAG: #define __SIZE_TYPE__ unsigned int
// LINUX32-DAG: #define __PTRDIFF_TYPE__ int
// LINUX32-DAG: #define __INTPTR_TYPE__ int
// LINUX32-DAG: #define __SIZEOF_SIZE_T__ 4
// LINUX32-DAG: #define __SIZEOF_PTRDIFF_T__ 4
// LINUX32-DAG: #define __SIZEOF_POINTER__ 4

// No host (SPIRV64 defaults)
// NOHOST64-DAG: #define __SIZE_TYPE__ long unsigned int
// NOHOST64-DAG: #define __PTRDIFF_TYPE__ long int
// NOHOST64-DAG: #define __INTPTR_TYPE__ long int
// NOHOST64-DAG: #define __SIZEOF_SIZE_T__ 8
// NOHOST64-DAG: #define __SIZEOF_PTRDIFF_T__ 8
// NOHOST64-DAG: #define __SIZEOF_POINTER__ 8

// No host (SPIRV32 defaults)
// NOHOST32-DAG: #define __SIZE_TYPE__ unsigned int
// NOHOST32-DAG: #define __PTRDIFF_TYPE__ int
// NOHOST32-DAG: #define __INTPTR_TYPE__ int
// NOHOST32-DAG: #define __SIZEOF_SIZE_T__ 4
// NOHOST32-DAG: #define __SIZEOF_PTRDIFF_T__ 4
// NOHOST32-DAG: #define __SIZEOF_POINTER__ 4

// Aux-target OS and arch macros
// WIN64-DAG: #define _WIN32 1
// WIN64-DAG: #define _WIN64 1
// WIN64-DAG: #define _M_X64 100
// WIN64-DAG: #define _M_AMD64 100
// LINUX64-DAG: #define __linux__ 1
// LINUX64-DAG: #define __x86_64__ 1

// SPIRV device macros always present
// LINUX64-DAG: #define __SPIRV__ 1
// LINUX64-DAG: #define __SPIRV64__ 1
// WIN64-DAG: #define __SPIRV__ 1
// WIN64-DAG: #define __SPIRV64__ 1
// NOHOST64-DAG: #define __SPIRV__ 1
// NOHOST64-DAG: #define __SPIRV64__ 1
// NOHOST32-DAG: #define __SPIRV__ 1
// NOHOST32-DAG: #define __SPIRV32__ 1
