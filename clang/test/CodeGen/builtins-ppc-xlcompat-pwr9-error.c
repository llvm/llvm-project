// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s

extern unsigned int ui;
extern unsigned long long ull;
extern long long ll;

void test_builtin_ppc_cmprb() {
  int res = __builtin_ppc_cmprb(3, ui, ui); // expected-error {{argument value 3 is outside the valid range [0, 1]}}
}

#ifdef __PPC64__

void test_builtin_ppc_addex() {
  long long res = __builtin_ppc_addex(ll, ll, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  unsigned long long res2 = __builtin_ppc_addex(ull, ull, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
}

#endif
