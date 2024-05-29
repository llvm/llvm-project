// RUN: rm -rf %t
// RUN: not clang-scan-deps --format=p1689 -- %clang this-file-does-not-exist.cpp 2>&1 | FileCheck %s --check-prefix=CHECK
// CHECK: error: no such file or directory: 'this-file-does-not-exist.cpp'
// CHECK: error: no input files
