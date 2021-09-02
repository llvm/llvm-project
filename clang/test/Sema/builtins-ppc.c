// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +htm                       \
// RUN: -triple powerpc64-unknown-unknown -DTEST_HTM -fsyntax-only      \
// RUN: -verify %s

// RUN: %clang_cc1 -target-feature +altivec -target-feature +crypto                    \
// RUN: -triple powerpc64le-unknown-unknown -DTEST_CRYPTO -fsyntax-only      \
// RUN: -verify %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +crypto \
// RUN: -triple powerpc64le-unknown-unknown -DTEST_CRYPTO -fsyntax-only \
// RUN: -target-feature +vsx -verify %s

#ifdef TEST_HTM
void test_htm() {
  __builtin_tbegin(4); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tend(-1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tsr(55);   // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tabortwc(-5, 2, 3); // expected-error-re {{argument value {{.*}} is outside the valid range}} 
  __builtin_tabortdc(55, 2, 3); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tabortwci(-5, 2, 5); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tabortwci(5, 2, 55); // expected-error-re {{argument value {{.*}} is outside the valid range}}  
  __builtin_tabortdci(-5, 2, 5); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_tabortdci(5, 2, 55); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}
#endif


#ifdef TEST_CRYPTO
#include <altivec.h>

#define W_INIT { 0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10 };
#define D_INIT { 0x0102030405060708, 0x090A0B0C0D0E0F10 };
vector unsigned int test_vshasigmaw_or(void)
{
  vector unsigned int a = W_INIT
  vector unsigned int b = __builtin_crypto_vshasigmaw(a, 2, 15);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned int c = __builtin_crypto_vshasigmaw(a, -1, 15); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned int d = __builtin_crypto_vshasigmaw(a, 0, 85);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned int e = __builtin_crypto_vshasigmaw(a, 1, -15); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_crypto_vshasigmaw(a, 1, 15);
}

#ifdef __VSX__
vector unsigned long long test_vshasigmad_or(void)
{
  vector unsigned long long a = D_INIT
  vector unsigned long long b = __builtin_crypto_vshasigmad(a, 2, 15);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned long long c = __builtin_crypto_vshasigmad(a, -1, 15); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned long long d = __builtin_crypto_vshasigmad(a, 0, 85);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned long long e = __builtin_crypto_vshasigmad(a, 1, -15); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_crypto_vshasigmad(a, 0, 15);
}
#endif

#endif

