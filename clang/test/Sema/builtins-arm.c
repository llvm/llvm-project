// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armv7 -target-abi apcs-gnu \
// RUN:   -fsyntax-only -verify %s

void f(void *a, void *b) {
  __clear_cache(); // expected-error {{too few arguments to function call, expected 2, have 0}} // expected-note {{'__clear_cache' is a builtin with type 'void (void *, void *)}}
  __clear_cache(a); // expected-error {{too few arguments to function call, expected 2, have 1}}
  __clear_cache(a, b);
}

void __clear_cache(char*, char*); // expected-error {{conflicting types for '__clear_cache'}}
void __clear_cache(void*, void*);

#if defined(__ARM_PCS) || defined(__ARM_EABI__)
// va_list on ARM AAPCS is struct { void* __ap }.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";
  *(ptr.__ap) = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}
#else
// va_list on ARM apcs-gnu is void*.
void test1() {
  __builtin_va_list ptr;
  ptr.__ap = "x";  // expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
  *(ptr.__ap) = '0';// expected-error {{member reference base type '__builtin_va_list' is not a structure or union}}
}

void test2() {
  __builtin_va_list ptr = "x";
  *ptr = '0'; // expected-error {{incomplete type 'void' is not assignable}}
}
#endif

void test3() {
  __builtin_arm_dsb(16); // expected-error {{argument should be a value from 0 to 15}}
  __builtin_arm_dmb(17); // expected-error {{argument should be a value from 0 to 15}}
  __builtin_arm_isb(18); // expected-error {{argument should be a value from 0 to 15}}
}

void test4() {
  __builtin_arm_prefetch(0, 2, 0); // expected-error {{argument should be a value from 0 to 1}}
  __builtin_arm_prefetch(0, 0, 2); // expected-error {{argument should be a value from 0 to 1}}
}

void test5() {
  __builtin_arm_dbg(16); // expected-error {{argument should be a value from 0 to 15}}
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
