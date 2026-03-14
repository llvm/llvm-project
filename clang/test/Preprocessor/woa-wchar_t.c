// RUN: %clang_cc1 -dM -triple armv7-windows -E %s | FileCheck %s
// RUN: %clang_cc1 -dM -fno-signed-char -triple armv7-windows -E %s | FileCheck %s

// CHECK: #define __WCHAR_TYPE__ unsigned short

