// RUN: %clang_cc1 -flax-vector-conversions=none -triple powerpc64le-linux-gnu -S -O0 -o - %s -target-feature +altivec -target-feature +vsx | FileCheck %s -check-prefix=CHECK -check-prefix=VSX
// RUN: %clang_cc1 -flax-vector-conversions=none -triple powerpc-linux-gnu -S -O0 -o - %s -target-feature +altivec -target-feature -vsx | FileCheck %s

// REQUIRES: powerpc-registered-target

#include <altivec.h>

// CHECK-LABEL: test1
// CHECK: vcfsx
vector float test1(vector int x) {
  return vec_ctf(x, 0);
}

// CHECK-LABEL: test2
// CHECK: vcfux
vector float test2(vector unsigned int x) {
  return vec_ctf(x, 0);
}

#ifdef __VSX__
// VSX-LABEL: test3
vector double test3(vector signed long long x) {
  return vec_ctf(x, 0);
}

// VSX-LABEL: test4
vector double test4(vector unsigned long long x) {
  return vec_ctf(x, 0);
}
#endif

// CHECK-LABEL: test5
// CHECK: vcfsx
vector float test5(vector int x) {
  return vec_vcfsx(x, 0);
}

// CHECK-LABEL: test6
// CHECK: vcfux
vector float test6(vector unsigned int x) {
  return vec_vcfux(x, 0);
}

// CHECK-LABEL: test7
// CHECK: vctsxs
vector int test7(vector float x) {
  return vec_cts(x, 0);
}

#ifdef __VSX__
// VSX-LABEL: test8
vector signed long long test8(vector double x) {
  return vec_cts(x, 0);
}

#endif

// CHECK-LABEL: test9
// CHECK: vctsxs
vector int test9(vector float x) {
  return vec_vctsxs(x, 0);
}

// CHECK-LABEL: test10
// CHECK: vctuxs
vector unsigned test10(vector float x) {
  return vec_ctu(x, 0);
}

#ifdef __VSX__
// VSX-LABEL: test11
vector unsigned long long test11(vector double x) {
  return vec_ctu(x, 0);
}

#endif

// CHECK-LABEL: test12
// CHECK: vctuxs
vector unsigned test12(vector float x) {
  return vec_vctuxs(x, 0);
}
