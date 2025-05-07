// RUN: not %clang -fbounds-safety -fexperimental-bounds-safety-cxx -### %s 2>&1 | FileCheck -check-prefixes T0 %s
// RUN: not %clang -fbounds-safety -fexperimental-bounds-safety-objc -### %s 2>&1 | FileCheck -check-prefixes T1 %s

// T0: error: unknown argument '-fexperimental-bounds-safety-cxx'; did you mean '-Xclang -fexperimental-bounds-safety-cxx'
// T1: error: unknown argument '-fexperimental-bounds-safety-objc'; did you mean '-Xclang -fexperimental-bounds-safety-objc'
