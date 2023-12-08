// RUN: %clang_cc1 -dM -triple armv7-windows -E %s | FileCheck %s
// RUN: %clang_cc1 -dM -fno-signed-char -triple armv7-windows -E %s \
// RUN:   | FileCheck %s -check-prefix CHECK-UNSIGNED-CHAR

// CHECK: #define _INTEGRAL_MAX_BITS 64
// CHECK: #define _M_ARM 7
// CHECK: #define _M_ARMT _M_ARM
// CHECK: #define _M_ARM_FP 31
// CHECK: #define _M_ARM_NT 1
// CHECK: #define _M_THUMB _M_ARM
// CHECK: #define _WIN32 1


// CHECK: #define __ARM_PCS 1
// CHECK: #define __ARM_PCS_VFP 1
// CHECK: #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
// CHECK: #define __INTPTR_TYPE__ int
// CHECK: #define __PTRDIFF_TYPE__ int
// CHECK: #define __SIZEOF_DOUBLE__ 8
// CHECK: #define __SIZEOF_FLOAT__ 4
// CHECK: #define __SIZEOF_INT__ 4
// CHECK: #define __SIZEOF_LONG_DOUBLE__ 8
// CHECK: #define __SIZEOF_LONG_LONG__ 8
// CHECK: #define __SIZEOF_LONG__ 4
// CHECK: #define __SIZEOF_POINTER__ 4
// CHECK: #define __SIZEOF_PTRDIFF_T__ 4
// CHECK: #define __SIZEOF_SHORT__ 2
// CHECK: #define __SIZEOF_SIZE_T__ 4
// CHECK: #define __SIZEOF_WCHAR_T__ 2
// CHECK: #define __SIZEOF_WINT_T__ 2
// CHECK: #define __SIZE_TYPE__ unsigned int
// CHECK: #define __UINTPTR_TYPE__ unsigned int

// CHECK-NOT: __THUMB_INTERWORK__
// CHECK-NOT: __ARM_EABI__
// CHECK-NOT: _CHAR_UNSIGNED

// CHECK-UNSIGNED-CHAR: #define _CHAR_UNSIGNED 1
