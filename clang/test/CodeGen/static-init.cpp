// RUN: not %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ %s \
// RUN: 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ %s \
// RUN: 2>&1 | FileCheck %s

struct test {
  test();
  ~test();
} t;

// CHECK: error in backend: Static initialization has not been implemented on XL ABI yet.
