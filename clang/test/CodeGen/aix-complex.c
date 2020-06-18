// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: fatal error: error in backend: complex type is not supported on AIX yet
_Complex float foo_float(_Complex float x) {
  return x;
}
