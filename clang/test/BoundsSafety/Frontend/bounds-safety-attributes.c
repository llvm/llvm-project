


// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -fsyntax-only %s

// RUN: %clang_cc1 -x c++ -fexperimental-bounds-safety-attributes -fsyntax-only %s

// RUN: %clang_cc1 -x objective-c -fexperimental-bounds-safety-attributes -fsyntax-only %s

// RUN: %clang_cc1 -x objective-c++ -fexperimental-bounds-safety-attributes -fsyntax-only %s

// RUN: %clang_cc1 -fbounds-safety -fexperimental-bounds-safety-attributes -fsyntax-only %s

// RUN: not %clang_cc1 -x c++ -fbounds-safety -fexperimental-bounds-safety-attributes -fsyntax-only %s 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -x objective-c -fbounds-safety -fexperimental-bounds-safety-attributes -fsyntax-only %s 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -x objective-c++ -fbounds-safety -fexperimental-bounds-safety-attributes -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error: -fbounds-safety is supported only for C language


// RUN: not %clang_cc1 -fbounds-safety -fno-experimental-bounds-safety-attributes -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-DIS %s

// CHECK-DIS: error: -fexperimental-bounds-safety-attributes cannot be disabled when -fbounds-safety is enabled
