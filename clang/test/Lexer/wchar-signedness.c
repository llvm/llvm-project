// RUN: %clang_cc1 -fsyntax-only -dM -E %s -triple x86_64-none-linux-gnu | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -fsyntax-only -dM -E %s -triple armv7-none-eabi | FileCheck %s --check-prefix=ARM

// CHECK-X86-NOT: #define __WCHAR_UNSIGNED__
// CHECK-X86: #define __WINT_UNSIGNED__ 1

// CHECK-ARM: #define __WCHAR_UNSIGNED__ 1
// CHECK-ARM-NOT: #define __WINT_UNSIGNED__ 1
