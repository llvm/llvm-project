// RUN: %clang --target=aarch64 %s -S -emit-llvm -o - | FileCheck %s --check-prefix=TC3
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature +float128 -DTEST_PPC128 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=IBM,PPC
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature +float128 -DTEST_PPC128 -mabi=ieeelongdouble -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=KC3,PPC

_Complex long double a, b, c, d;
void test_fp128_compound_assign(void) {
  // TC3: call { fp128, fp128 } @__multc3
  // IBM: call { ppc_fp128, ppc_fp128 } @__multc3
  // KC3: call { fp128, fp128 } @__mulkc3
  a *= b;
  // TC3: call { fp128, fp128 } @__divtc3
  // IBM: call { ppc_fp128, ppc_fp128 } @__divtc3
  // KC3: call { fp128, fp128 } @__divkc3
  c /= d;
}

#ifdef __FLOAT128__
_Complex __float128 e, f, g, h;
void test_float128_compound_assign(void) {
  // PPC: call { fp128, fp128 } @__mulkc3
  e *= f;
  // PPC: call { fp128, fp128 } @__divkc3
  g /= h;
}
#endif

#ifdef __powerpc__
_Complex __ibm128 i, j, k, l;
void test_ibm128_compound_assign(void) {
  // PPC: call { ppc_fp128, ppc_fp128 } @__multc3
  i *= j;
  // PPC: call { ppc_fp128, ppc_fp128 } @__divtc3
  k /= l;
}
#endif
