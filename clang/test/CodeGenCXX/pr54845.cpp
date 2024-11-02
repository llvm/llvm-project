// RUN: %clang_cc1 -triple i686-linux-gnu -emit-llvm %s -o - | FileCheck %s
// https://github.com/llvm/llvm-project/issues/54845

void *operator new(unsigned int, void *);

void test(double *d) {
  // This store used to have an alignment of 8, which was incorrect as
  // the i386 psABI only guarantees a 4-byte alignment for doubles.

  // CHECK: store double 0.000000e+00, {{.*}}, align 4
  new (d) double(0);
}
