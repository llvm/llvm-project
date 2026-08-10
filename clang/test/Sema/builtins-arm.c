// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armv7 -target-abi apcs-gnu \
// RUN:   -fsyntax-only -verify %s

#include <arm_acle.h>

void f(void *a, void *b) {
  __clear_cache(); // expected-error {{too few arguments to function call, expected 2, have 0}} // expected-note {{'__clear_cache' is a builtin with type 'void (void *, void *)}}
  __clear_cache(a); // expected-error {{too few arguments to function call, expected 2, have 1}}
  __clear_cache(a, b);
}

void __clear_cache(char*, char*); // expected-error {{conflicting types for '__clear_cache'}}
void __clear_cache(void*, void*);

#if defined(__ARM_PCS) || defined(__ARM_EABI__)
// va_list on ARM AAPCS is struct { void* __ap }.
void test1(void) {
  __builtin_va_list ptr;
  ptr.__ap = "x";
  *(ptr.__ap) = '0'; /* expected-error {{incomplete type 'void' is not assignable}}
                        expected-warning {{ISO C does not allow indirection on operand of type 'void *'}} */
}
#else
// va_list on ARM apcs-gnu is void*.
void test1(void) {
  __builtin_va_list ptr;
  ptr.__ap = "x";  // expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
  *(ptr.__ap) = '0';// expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
}

void test2(void) {
  __builtin_va_list ptr = "x";
  *ptr = '0'; /* expected-error {{incomplete type 'void' is not assignable}}
                 expected-warning {{ISO C does not allow indirection on operand of type '__builtin_va_list'}} */

}
#endif

void test3(void) {
  __builtin_arm_dsb(16); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_dmb(17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_isb(18); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test4(void) {
  __builtin_arm_prefetch(0, 2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_prefetch(0, 0, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test5(void) {
  __builtin_arm_dbg(16); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test6(int a, int b, int c) {
  __builtin_arm_ldc(1, 2, &a);
  __builtin_arm_ldc(a, 2, &a); // expected-error {{argument to '__builtin_arm_ldc' must be a constant integer}}
  __builtin_arm_ldc(1, a, &a); // expected-error {{argument to '__builtin_arm_ldc' must be a constant integer}}

  __builtin_arm_ldcl(1, 2, &a);
  __builtin_arm_ldcl(a, 2, &a); // expected-error {{argument to '__builtin_arm_ldcl' must be a constant integer}}
  __builtin_arm_ldcl(1, a, &a); // expected-error {{argument to '__builtin_arm_ldcl' must be a constant integer}}

  __builtin_arm_ldc2(1, 2, &a);
  __builtin_arm_ldc2(a, 2, &a); // expected-error {{argument to '__builtin_arm_ldc2' must be a constant integer}}
  __builtin_arm_ldc2(1, a, &a); // expected-error {{argument to '__builtin_arm_ldc2' must be a constant integer}}

  __builtin_arm_ldc2l(1, 2, &a);
  __builtin_arm_ldc2l(a, 2, &a); // expected-error {{argument to '__builtin_arm_ldc2l' must be a constant integer}}
  __builtin_arm_ldc2l(1, a, &a); // expected-error {{argument to '__builtin_arm_ldc2l' must be a constant integer}}

  __builtin_arm_stc(1, 2, &a);
  __builtin_arm_stc(a, 2, &a); // expected-error {{argument to '__builtin_arm_stc' must be a constant integer}}
  __builtin_arm_stc(1, a, &a); // expected-error {{argument to '__builtin_arm_stc' must be a constant integer}}

  __builtin_arm_stcl(1, 2, &a);
  __builtin_arm_stcl(a, 2, &a); // expected-error {{argument to '__builtin_arm_stcl' must be a constant integer}}
  __builtin_arm_stcl(1, a, &a); // expected-error {{argument to '__builtin_arm_stcl' must be a constant integer}}

  __builtin_arm_stc2(1, 2, &a);
  __builtin_arm_stc2(a, 2, &a); // expected-error {{argument to '__builtin_arm_stc2' must be a constant integer}}
  __builtin_arm_stc2(1, a, &a); // expected-error {{argument to '__builtin_arm_stc2' must be a constant integer}}

  __builtin_arm_stc2l(1, 2, &a);
  __builtin_arm_stc2l(a, 2, &a); // expected-error {{argument to '__builtin_arm_stc2l' must be a constant integer}}
  __builtin_arm_stc2l(1, a, &a); // expected-error {{argument to '__builtin_arm_stc2l' must be a constant integer}}

  __builtin_arm_cdp(a, 2, 3, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp' must be a constant integer}}
  __builtin_arm_cdp(1, a, 3, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp' must be a constant integer}}
  __builtin_arm_cdp(1, 2, a, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp' must be a constant integer}}
  __builtin_arm_cdp(1, 2, 3, a, 5, 6); // expected-error {{argument to '__builtin_arm_cdp' must be a constant integer}}
  __builtin_arm_cdp(1, 2, 3, 4, 5, a); // expected-error {{argument to '__builtin_arm_cdp' must be a constant integer}}

  __builtin_arm_cdp2(a, 2, 3, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp2' must be a constant integer}}
  __builtin_arm_cdp2(1, a, 3, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp2' must be a constant integer}}
  __builtin_arm_cdp2(1, 2, a, 4, 5, 6); // expected-error {{argument to '__builtin_arm_cdp2' must be a constant integer}}
  __builtin_arm_cdp2(1, 2, 3, a, 5, 6); // expected-error {{argument to '__builtin_arm_cdp2' must be a constant integer}}
  __builtin_arm_cdp2(1, 2, 3, 4, 5, a); // expected-error {{argument to '__builtin_arm_cdp2' must be a constant integer}}

  __builtin_arm_mrc( a, 0, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mrc' must be a constant integer}}
  __builtin_arm_mrc(15, a, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mrc' must be a constant integer}}
  __builtin_arm_mrc(15, 0,  a, 0, 3); // expected-error {{argument to '__builtin_arm_mrc' must be a constant integer}}
  __builtin_arm_mrc(15, 0, 13, a, 3); // expected-error {{argument to '__builtin_arm_mrc' must be a constant integer}}
  __builtin_arm_mrc(15, 0, 13, 0, a); // expected-error {{argument to '__builtin_arm_mrc' must be a constant integer}}

  __builtin_arm_mrc2( a, 0, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mrc2' must be a constant integer}}
  __builtin_arm_mrc2(15, a, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mrc2' must be a constant integer}}
  __builtin_arm_mrc2(15, 0,  a, 0, 3); // expected-error {{argument to '__builtin_arm_mrc2' must be a constant integer}}
  __builtin_arm_mrc2(15, 0, 13, a, 3); // expected-error {{argument to '__builtin_arm_mrc2' must be a constant integer}}
  __builtin_arm_mrc2(15, 0, 13, 0, a); // expected-error {{argument to '__builtin_arm_mrc2' must be a constant integer}}

  __builtin_arm_mcr( a, 0, b, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mcr' must be a constant integer}}
  __builtin_arm_mcr(15, a, b, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mcr' must be a constant integer}}
  __builtin_arm_mcr(15, 0, b,  a, 0, 3); // expected-error {{argument to '__builtin_arm_mcr' must be a constant integer}}
  __builtin_arm_mcr(15, 0, b, 13, a, 3); // expected-error {{argument to '__builtin_arm_mcr' must be a constant integer}}
  __builtin_arm_mcr(15, 0, b, 13, 0, a); // expected-error {{argument to '__builtin_arm_mcr' must be a constant integer}}

  __builtin_arm_mcr2( a, 0, b, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mcr2' must be a constant integer}}
  __builtin_arm_mcr2(15, a, b, 13, 0, 3); // expected-error {{argument to '__builtin_arm_mcr2' must be a constant integer}}
  __builtin_arm_mcr2(15, 0, b,  a, 0, 3); // expected-error {{argument to '__builtin_arm_mcr2' must be a constant integer}}
  __builtin_arm_mcr2(15, 0, b, 13, a, 3); // expected-error {{argument to '__builtin_arm_mcr2' must be a constant integer}}
  __builtin_arm_mcr2(15, 0, b, 13, 0, a); // expected-error {{argument to '__builtin_arm_mcr2' must be a constant integer}}

  __builtin_arm_mcrr(15, 0, b, 0);
  __builtin_arm_mcrr( a, 0, b, 0); // expected-error {{argument to '__builtin_arm_mcrr' must be a constant integer}}
  __builtin_arm_mcrr(15, a, b, 0); // expected-error {{argument to '__builtin_arm_mcrr' must be a constant integer}}
  __builtin_arm_mcrr(15, 0, b, a); // expected-error {{argument to '__builtin_arm_mcrr' must be a constant integer}}

  __builtin_arm_mcrr2(15, 0, b, 0);
  __builtin_arm_mcrr2( a, 0, b, 0); // expected-error {{argument to '__builtin_arm_mcrr2' must be a constant integer}}
  __builtin_arm_mcrr2(15, a, b, 0); // expected-error {{argument to '__builtin_arm_mcrr2' must be a constant integer}}
  __builtin_arm_mcrr2(15, 0, b, a); // expected-error {{argument to '__builtin_arm_mcrr2' must be a constant integer}}

  __builtin_arm_mrrc(15, 0, 0);
  __builtin_arm_mrrc( a, 0, 0); // expected-error {{argument to '__builtin_arm_mrrc' must be a constant integer}}
  __builtin_arm_mrrc(15, a, 0); // expected-error {{argument to '__builtin_arm_mrrc' must be a constant integer}}
  __builtin_arm_mrrc(15, 0, a); // expected-error {{argument to '__builtin_arm_mrrc' must be a constant integer}}

  __builtin_arm_mrrc2(15, 0, 0);
  __builtin_arm_mrrc2( a, 0, 0); // expected-error {{argument to '__builtin_arm_mrrc2' must be a constant integer}}
  __builtin_arm_mrrc2(15, a, 0); // expected-error {{argument to '__builtin_arm_mrrc2' must be a constant integer}}
  __builtin_arm_mrrc2(15, 0, a); // expected-error {{argument to '__builtin_arm_mrrc2' must be a constant integer}}
}

void test_9_3_multiplications(int a, int b) {
  int r;
  r = __builtin_arm_smulbb(a, b);
  r = __builtin_arm_smulbb(1, -9);

  r = __builtin_arm_smulbt(a, b);
  r = __builtin_arm_smulbt(0, b);

  r = __builtin_arm_smultb(a, b);
  r = __builtin_arm_smultb(5, b);

  r = __builtin_arm_smultt(a, b);
  r = __builtin_arm_smultt(a, -1);

  r = __builtin_arm_smulwb(a, b);
  r = __builtin_arm_smulwb(1, 2);

  r = __builtin_arm_smulwt(a, b);
  r = __builtin_arm_smulwt(-1, -2);
  r = __builtin_arm_smulwt(-1.0f, -2);
}

void test_9_4_1_width_specified_saturation(int a, int b) {
  unsigned u;
  int s;

  s = __builtin_arm_ssat(8, 2);
  s = __builtin_arm_ssat(a, 1);
  s = __builtin_arm_ssat(a, 32);
  s = __builtin_arm_ssat(a, 0);   // expected-error-re {{argument value {{.*}} is outside the valid range}}
  s = __builtin_arm_ssat(a, 33);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  s = __builtin_arm_ssat(a, b);   // expected-error {{argument to '__builtin_arm_ssat' must be a constant integer}}

  u = __builtin_arm_usat(8, 2);
  u = __builtin_arm_usat(a, 0);
  u = __builtin_arm_usat(a, 31);
  u = __builtin_arm_usat(a, 32);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  u = __builtin_arm_usat(a, b);   // expected-error {{argument to '__builtin_arm_usat' must be a constant integer}}
}

void test_9_4_2_saturating_addition_subtraction(int a, int b) {
  int s;
  s = __builtin_arm_qadd(a, b);
  s = __builtin_arm_qadd(-1, 0);

  s = __builtin_arm_qsub(a, b);
  s = __builtin_arm_qsub(0, -1);

  s = __builtin_arm_qdbl(a);
}

void test_9_4_3_accumulating_multiplications(int a, int b, int c) {
  int s;

  s = __builtin_arm_smlabb(a, b, c);
  s = __builtin_arm_smlabb(1, b, c);
  s = __builtin_arm_smlabb(a, 2, c);
  s = __builtin_arm_smlabb(a, b, -3);

  s = __builtin_arm_smlabt(a, b, c);
  s = __builtin_arm_smlabt(1, b, c);
  s = __builtin_arm_smlabt(a, 2, c);
  s = __builtin_arm_smlabt(a, b, -3);

  s = __builtin_arm_smlatb(a, b, c);
  s = __builtin_arm_smlatt(1, b, c);
  s = __builtin_arm_smlawb(a, 2, c);
  s = __builtin_arm_smlawt(a, b, -3);
}

void test_9_5_4_parallel_16bit_saturation(int16x2_t a) {
  unsigned u;
  int s;

  s = __builtin_arm_ssat16(a, 1);
  s = __builtin_arm_ssat16(a, 16);
  s = __builtin_arm_ssat16(a, 0);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  s = __builtin_arm_ssat16(a, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  u = __builtin_arm_usat16(a, 0);
  u = __builtin_arm_usat16(a, 15);
  u = __builtin_arm_usat16(a, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_9_5_5_packing_and_unpacking(int16x2_t a, int8x4_t b, uint16x2_t c, uint8x4_t d) {
  int16x2_t x;
  uint16x2_t y;

  x = __builtin_arm_sxtab16(a, b);
  x = __builtin_arm_sxtab16(1, -1);
  x = __builtin_arm_sxtb16(b);
  x = __builtin_arm_sxtb16(-b);

  y = __builtin_arm_uxtab16(c, d);
  y = __builtin_arm_uxtab16(-1, -2);
  y = __builtin_arm_uxtb16(d);
  y = __builtin_arm_uxtb16(-1);
}

uint8x4_t
test_9_5_6_parallel_selection(uint8x4_t a, uint8x4_t b) {
  return __builtin_arm_sel(a, b);
}

void test_9_5_7_parallel_8bit_addition_substraction(int8x4_t a, int8x4_t b,
                                                    uint8x4_t c, uint8x4_t d) {
  int8x4_t s;
  uint8x4_t u;

  s = __builtin_arm_qadd8(a, b);
  s = __builtin_arm_qsub8(a, b);
  s = __builtin_arm_sadd8(a, b);
  s = __builtin_arm_shadd8(a, b);
  s = __builtin_arm_shsub8(a, b);
  s = __builtin_arm_ssub8(a, b);

  u = __builtin_arm_uadd8(c, d);
  u = __builtin_arm_uhadd8(c, d);
  u = __builtin_arm_uhsub8(c, d);
  u = __builtin_arm_uqadd8(c, d);
  u = __builtin_arm_uqsub8(c, d);
  u = __builtin_arm_usub8(c, d);
}

void test_9_5_8_absolute_differences(uint8x4_t a, uint8x4_t b, uint32_t c) {
  uint32_t r;

  r = __builtin_arm_usad8(a, b);
  r = __builtin_arm_usada8(a, b, c);
}

void test_9_5_9_parallel_addition_and_subtraction(int16x2_t a, int16x2_t b,
                                                  uint16x2_t c, uint16x2_t d) {
  int16x2_t x;
  uint16x2_t y;

  x = __builtin_arm_qadd16(a, b);
  x = __builtin_arm_qasx(a, b);
  x = __builtin_arm_qsax(a, b);
  x = __builtin_arm_qsub16(a, b);
  x = __builtin_arm_sadd16(a, b);
  x = __builtin_arm_sasx(a, b);
  x = __builtin_arm_shadd16(a, b);
  x = __builtin_arm_shasx(a, b);
  x = __builtin_arm_shsax(a, b);
  x = __builtin_arm_shsub16(a, b);
  x = __builtin_arm_ssax(a, b);
  x = __builtin_arm_ssub16(a, b);

  y = __builtin_arm_uadd16(c, d);
  y = __builtin_arm_uasx(c, d);
  y = __builtin_arm_uhadd16(c, d);
  y = __builtin_arm_uhasx(c, d);
  y = __builtin_arm_uhsax(c, d);
  y = __builtin_arm_uhsub16(c, d);
  y = __builtin_arm_uqadd16(c, d);
  y = __builtin_arm_uqasx(c, d);
  y = __builtin_arm_uqsax(c, d);
  y = __builtin_arm_uqsub16(c, d);
  y = __builtin_arm_usax(c, d);
  y = __builtin_arm_usub16(c, d);
}

void test_9_5_10_parallel_16bit_multiplication(int16x2_t a, int16x2_t b,
                                               int32_t c, int64_t d) {
  int32_t x;
  int64_t y;

  x = __builtin_arm_smlad(a, b, c);
  x = __builtin_arm_smladx(a, b, c);
  y = __builtin_arm_smlald(a, b, d);
  y = __builtin_arm_smlaldx(a, b, d);
  x = __builtin_arm_smlsd(a, b, c);
  x = __builtin_arm_smlsdx(a, b, c);
  y = __builtin_arm_smlsld(a, b, d);
  y = __builtin_arm_smlsldx(a, b, d);
  x = __builtin_arm_smuad(a, b);
  x = __builtin_arm_smuadx(a, b);
  x = __builtin_arm_smusd(a, b);
  x = __builtin_arm_smusdx(a, b);
}

void test_VFP(float f, double d) {
  float fr;
  double dr;

  fr = __builtin_arm_vcvtr_f(f, 0);
  fr = __builtin_arm_vcvtr_f(f, 1);
  fr = __builtin_arm_vcvtr_f(f, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  fr = __builtin_arm_vcvtr_f(f, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}

  dr = __builtin_arm_vcvtr_f(d, 0);
  dr = __builtin_arm_vcvtr_f(d, 1);
  dr = __builtin_arm_vcvtr_f(d, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  dr = __builtin_arm_vcvtr_f(d, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}
