// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple powerpc-unknown-aix  -target-feature +altivec \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix  -target-feature +altivec \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: fatal error: error in backend: vector type is not supported on AIX yet
vector signed int retVector(vector signed int x) {
  return x;
}
