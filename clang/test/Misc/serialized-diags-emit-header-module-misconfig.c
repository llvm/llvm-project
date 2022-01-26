// RUN: rm -rf %t && mkdir %t
// RUN: not %clang_cc1 -emit-header-module %s -o %t/out.pcm -serialize-diagnostic-file %t/diag 2>&1 | FileCheck %s

// CHECK: error: header module compilation requires '-fmodules', '-std=c++20', or '-fmodules-ts'
