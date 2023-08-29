// RUN: %clang_cc1 -fsyntax-only -E -dM -triple=armv7k-apple-watchos %s | FileCheck --check-prefix=BOOL %s
// RUN: %clang_cc1 -fsyntax-only -E -dM -triple=x86_64-apple-darwin16 %s | FileCheck --check-prefix=CHAR %s
// RUN: %clang_cc1 -x c -fsyntax-only -E -dM -triple=x86_64-apple-darwin16 %s | FileCheck --check-prefix=CHAR %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -E -dM -triple=x86_64-apple-darwin16 %s | FileCheck --check-prefix=CHAR %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -E -dM -triple=x86_64-apple-darwin16 %s | FileCheck --check-prefix=CHAR %s

// BOOL: #define __OBJC_BOOL_IS_BOOL 1
// BOOL-NOT: #define __OBJC_BOOL_IS_BOOL 0

// CHAR: #define __OBJC_BOOL_IS_BOOL 0
// CHAR-NOT: #define __OBJC_BOOL_IS_BOOL 1
