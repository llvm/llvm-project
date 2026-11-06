// RUN: %clang_cc1 -E -dM -ffreestanding -triple=powerpc64-unknown-xbox360 < /dev/null | FileCheck -match-full-lines %s
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -triple=powerpc64-unknown-xbox360 %s

// CHECK-DAG: #define _M_PPC 5401
// CHECK-DAG: #define _M_PPCBE 1
// CHECK-DAG: #define _WIN32 1
// CHECK-DAG: #define _XBOX_VER 200

// CHECK-DAG: #define __LDBL_MANT_DIG__ 53
// CHECK-DAG: #define __SIZEOF_LONG_DOUBLE__ 8
// CHECK-DAG: #define __SIZEOF_WCHAR_T__ 2
// CHECK-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-DAG: #define __WCHAR_UNSIGNED__ 1

_Static_assert(sizeof(long double) == 8, "");
_Static_assert(_Alignof(long double) == 8, "");
_Static_assert(sizeof(__WCHAR_TYPE__) == 2, "");
_Static_assert((__WCHAR_TYPE__)-1 > 0, "");
