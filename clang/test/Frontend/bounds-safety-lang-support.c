// RUN: not %clang_cc1 -fexperimental-bounds-safety -x c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang_cc1 -fexperimental-bounds-safety -x objective-c %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang_cc1 -fexperimental-bounds-safety -x objective-c++ %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang_cc1 -fexperimental-bounds-safety -x cuda -nocudalib -nocudainc %s 2>&1 | FileCheck -check-prefix ERR %s
// RUN: not %clang_cc1 -fexperimental-bounds-safety -x renderscript %s 2>&1 | FileCheck -check-prefix ERR %s

// ERR: error: '-fexperimental-bounds-safety' is only supported for C
