// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr9 \
// RUN:   -verify %s

extern unsigned long long ull;
extern long long ll;

void test_builtin_ppc_addex() {
  long long res = __builtin_ppc_addex(ll, ll, 1); // expected-warning {{argument value 1 will result in undefined behaviour}}
  unsigned long long res2 = __builtin_ppc_addex(ull, ull, 3); // expected-warning {{argument value 3 will result in undefined behaviour}}
}
