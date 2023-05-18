// RUN: %clang_cc1 -triple armv8 -target-cpu cortex-a57 -fsyntax-only -ffreestanding -verify %s

#include <arm_acle.h>
/*
 * Memory barrier intrinsics
 * Argument for dmb, dsb, isb must be compile-time constant,
 * otherwise an error should be raised.
 */
void test_dmb_const_diag(const unsigned int t) {
  return __dmb(t);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

void test_dsb_const_diag(const unsigned int t) {
  return __dsb(t);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

void test_isb_const_diag(const unsigned int t) {
  return __isb(t);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

/*
 * Saturating intrinsics
 * Second argument for SSAT and USAT intrinsics must be compile-time constant,
 * otherwise an error should be raised.
 */
int32_t test_ssat_const_diag(int32_t t, const int32_t v) {
  return __ssat(t, v);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

int32_t test_usat_const_diag(int32_t t, const int32_t v) {
  return __usat(t, v);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

/*
 * Prefetch intrinsics
 */
void test_pldx_const_diag(int32_t i) {
  __pldx(i, 0, 0, 0);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

/*
 * DBG intrinsic
 * First argument for DBG intrinsic must be compile-time constant,
 * otherwise an error should be raised.
 */
void test_dbg_const_diag(unsigned int t) {
  __dbg(t);  // expected-error-re {{argument to {{.*}} must be a constant integer}}
}
