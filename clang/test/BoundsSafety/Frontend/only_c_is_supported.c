// RUN: not %clang -fbounds-safety-experimental -x c++ %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang -fbounds-safety-experimental -x objective-c %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang -fbounds-safety-experimental -x objective-c++ %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang -fbounds-safety-experimental -x cuda -nocudalib -nocudainc %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang -fbounds-safety-experimental -x renderscript %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang_cc1 -fbounds-safety-experimental -x c++ %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang_cc1 -fbounds-safety-experimental -x objective-c %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang_cc1 -fbounds-safety-experimental -x objective-c++ %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang_cc1 -fbounds-safety-experimental -x cuda %s 2>&1 | FileCheck -check-prefix ERR %s

// RUN: not %clang_cc1 -fbounds-safety-experimental -x renderscript %s 2>&1 | FileCheck -check-prefix ERR %s

// ERR: error: bounds safety is only supported for C

// expected-no-diagnostics
// RUN: %clang -fbounds-safety-experimental -fsyntax-only -Xclang -verify -c -x c %s
// RUN: %clang_cc1 -fbounds-safety-experimental -fsyntax-only -verify -x c %s