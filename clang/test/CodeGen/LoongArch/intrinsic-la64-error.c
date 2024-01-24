// RUN: %clang_cc1 -triple loongarch64 -emit-llvm -S -verify %s -o /dev/null
// RUN: not %clang_cc1 -triple loongarch64 -DFEATURE_CHECK -emit-llvm %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

#include <larchintrin.h>

#ifdef FEATURE_CHECK
void test_feature(unsigned long *v_ul, int *v_i, float a, double b) {
// CHECK: error: '__builtin_loongarch_cacop_w' needs target feature 32bit
  __builtin_loongarch_cacop_w(1, v_ul[0], 1024);
// CHECK: error: '__builtin_loongarch_movfcsr2gr' needs target feature f
  v_i[0] = __builtin_loongarch_movfcsr2gr(1);
// CHECK: error: '__builtin_loongarch_movgr2fcsr' needs target feature f
  __builtin_loongarch_movgr2fcsr(1, v_i[1]);
// CHECK: error: '__builtin_loongarch_frecipe_s' needs target feature f,frecipe
  float f1 = __builtin_loongarch_frecipe_s(a);
// CHECK: error: '__builtin_loongarch_frsqrte_s' needs target feature f,frecipe
  float f2 = __builtin_loongarch_frsqrte_s(a);
// CHECK: error: '__builtin_loongarch_frecipe_d' needs target feature d,frecipe
  double d1 = __builtin_loongarch_frecipe_d(b);
// CHECK: error: '__builtin_loongarch_frsqrte_d' needs target feature d,frecipe
  double d2 = __builtin_loongarch_frsqrte_d(b);
}
#endif

void csrrd_d(int a) {
  __builtin_loongarch_csrrd_d(16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrrd_d(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrrd_d(a); // expected-error {{argument to '__builtin_loongarch_csrrd_d' must be a constant integer}}
}

void csrwr_d(unsigned long int a) {
  __builtin_loongarch_csrwr_d(a, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrwr_d(a, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrwr_d(a, a); // expected-error {{argument to '__builtin_loongarch_csrwr_d' must be a constant integer}}
}

void csrxchg_d(unsigned long int a, unsigned long int b) {
  __builtin_loongarch_csrxchg_d(a, b, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrxchg_d(a, b, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrxchg_d(a, b, b); // expected-error {{argument to '__builtin_loongarch_csrxchg_d' must be a constant integer}}
}

void lddir_d(long int a, int b) {
  __builtin_loongarch_lddir_d(a, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_loongarch_lddir_d(a, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 31]}}
  __builtin_loongarch_lddir_d(a, b); // expected-error {{argument to '__builtin_loongarch_lddir_d' must be a constant integer}}
}

void ldpte_d(long int a, int b) {
  __builtin_loongarch_ldpte_d(a, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_loongarch_ldpte_d(a, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 31]}}
  __builtin_loongarch_ldpte_d(a, b); // expected-error {{argument to '__builtin_loongarch_ldpte_d' must be a constant integer}}
}

int movfcsr2gr_out_of_lo_range(int a) {
  int b = __builtin_loongarch_movfcsr2gr(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  int c = __builtin_loongarch_movfcsr2gr(32); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  int d = __builtin_loongarch_movfcsr2gr(a); // expected-error {{argument to '__builtin_loongarch_movfcsr2gr' must be a constant integer}}
  return 0;
}

void movgr2fcsr(int a, int b) {
  __builtin_loongarch_movgr2fcsr(-1, b); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(32, b); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(a, b); // expected-error {{argument to '__builtin_loongarch_movgr2fcsr' must be a constant integer}}
}
