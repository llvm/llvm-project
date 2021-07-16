// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fsyntax-only \
// RUN:   -Wall -Werror -verify %s

extern long long lla, llb;
extern int ia, ib;
extern unsigned int ui;
extern unsigned long long ull;

void test_trap(void) {
#ifdef __PPC64__
  __tdw(lla, llb, 50); //expected-error {{argument value 50 is outside the valid range [0, 31]}}
#endif
  __tw(ia, ib, 50); //expected-error {{argument value 50 is outside the valid range [0, 31]}}
}

void test_builtin_ppc_rldimi() {
  unsigned int shift;
  unsigned long long mask;
  unsigned long long res = __builtin_ppc_rldimi(ull, ull, shift, 7); // expected-error {{argument to '__builtin_ppc_rldimi' must be a constant integer}}
  res = __builtin_ppc_rldimi(ull, ull, 63, mask);                    // expected-error {{argument to '__builtin_ppc_rldimi' must be a constant integer}}
  res = __builtin_ppc_rldimi(ull, ull, 63, 0xFFFF000000000F00);      // expected-error {{argument 3 value should represent a contiguous bit field}}
}

void test_builtin_ppc_rlwimi() {
  unsigned int shift;
  unsigned int mask;
  unsigned int res = __builtin_ppc_rlwimi(ui, ui, shift, 7); // expected-error {{argument to '__builtin_ppc_rlwimi' must be a constant integer}}
  res = __builtin_ppc_rlwimi(ui, ui, 31, mask);              // expected-error {{argument to '__builtin_ppc_rlwimi' must be a constant integer}}
  res = __builtin_ppc_rlwimi(ui, ui, 31, 0xFFFF0F00);        // expected-error {{argument 3 value should represent a contiguous bit field}}
}

void test_builtin_ppc_rlwnm() {
  unsigned int shift;
  unsigned int mask;
  unsigned int res = __builtin_ppc_rlwnm(ui, shift, 7); // expected-error {{argument to '__builtin_ppc_rlwnm' must be a constant integer}}
  res = __builtin_ppc_rlwnm(ui, 31, mask);              // expected-error {{argument to '__builtin_ppc_rlwnm' must be a constant integer}}
  res = __builtin_ppc_rlwnm(ui, 31, 0xFF0F0F00);        // expected-error {{argument 2 value should represent a contiguous bit field}}
}
