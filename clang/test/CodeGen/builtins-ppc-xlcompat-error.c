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

extern unsigned int usi;
extern double d;
extern float f;

void testMathBuiltin(void) {
  __mtfsb0(usi); //expected-error {{argument to '__builtin_ppc_mtfsb0' must be a constant integer}}
  __mtfsb0(32); //expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __mtfsb1(usi); //expected-error {{argument to '__builtin_ppc_mtfsb1' must be a constant integer}}
  __mtfsb1(45); //expected-error {{argument value 45 is outside the valid range [0, 31]}}
  __mtfsf(usi, usi); //expected-error {{argument to '__builtin_ppc_mtfsf' must be a constant integer}}
  __mtfsf(350, usi); //expected-error {{argument value 350 is outside the valid range [0, 255]}}
  __mtfsfi(usi, 0); //expected-error {{argument to '__builtin_ppc_mtfsfi' must be a constant integer}}
  __mtfsfi(0, usi); //expected-error {{argument to '__builtin_ppc_mtfsfi' must be a constant integer}}
  __mtfsfi(8, 0); //expected-error {{argument value 8 is outside the valid range [0, 7]}}
  __mtfsfi(5, 24); //expected-error {{argument value 24 is outside the valid range [0, 15]}}
}
