// RUN: %clang_cc1 -flax-vector-conversions=none -triple powerpc-linux-gnu -S -O0 -o - %s -target-feature +altivec | FileCheck %s

// REQUIRES: powerpc-registered-target

#include <altivec.h>

// CHECK-LABEL: test1
// CHECK: dss 
void test1() {
  vec_dss(1);
}
