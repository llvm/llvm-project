// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -DONLY_GOOD %s -emit-llvm -o - | FileCheck %s --check-prefix=GOOD

#ifndef ONLY_GOOD
struct Big {
  unsigned long long a[128];
};

struct Big callee(struct Big x);
struct Big caller(struct Big x) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  __attribute__((musttail)) return callee(x);
}
#endif

struct Small {
  int a[4];
};
struct Small small_callee(struct Small x);
struct Small small_caller(struct Small x) {
  __attribute__((musttail)) return small_callee(x);
}
// GOOD-LABEL: define {{.*}} @small_caller(
// GOOD: musttail call {{.*}} @small_callee(
