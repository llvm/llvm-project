// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -O2 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef float v4sf __attribute__ ((vector_size (16)));

struct s { v4sf v; };

v4sf foo (struct s a) {
  return a.v;
}

// CHECK-LABEL: define <4 x float> @foo(<4 x float> inreg %a.coerce)
// CHECK: ret <4 x float> %a.coerce
