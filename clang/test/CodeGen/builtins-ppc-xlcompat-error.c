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
extern const int cia;
extern unsigned long ula;

void test_trap(void) {
#ifdef __PPC64__
  __tdw(lla, llb, 50); //expected-error {{argument value 50 is outside the valid range [1, 31]}}
  __tdw(lla, llb, 0); //expected-error {{argument value 0 is outside the valid range [1, 31]}}
#endif
  __tw(ia, ib, 50); //expected-error {{argument value 50 is outside the valid range [1, 31]}}
  __tw(ia, ib, 0); //expected-error {{argument value 0 is outside the valid range [1, 31]}}
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

unsigned long long testrdlam(unsigned long long rs, unsigned int shift, unsigned int not_const) {
  // The third parameter is a mask that must be a constant that represents a
  // contiguous bit field.
  unsigned long long Return;
  // Third parameter is not a constant.
  Return = __rdlam(rs, shift, not_const); //expected-error {{argument to '__builtin_ppc_rdlam' must be a constant integer}}
  // Third parameter is a constant but not a contiguous bit field.
  return __rdlam(rs, shift, 0xF4) + Return; //expected-error {{argument 2 value should represent a contiguous bit field}}
}

void testalignx(const void *pointer, unsigned int alignment) {
  // The alignment must be an immediate.
  __alignx(alignment, pointer); //expected-error {{argument to '__builtin_ppc_alignx' must be a constant integer}}
  // The alignment must be a power of 2.
  __alignx(0x0, pointer); //expected-error {{argument should be a power of 2}}
  // The alignment must be a power of 2.
  __alignx(0xFF, pointer); //expected-error {{argument should be a power of 2}}
}

#ifndef __PPC64__
long long testbpermd(long long bit_selector, long long source) {
  return __bpermd(bit_selector, source); //expected-error {{this builtin is only available on 64-bit targets}}
}

long long testdivde(long long dividend, long long divisor) {
  return __divde(dividend, divisor); //expected-error {{this builtin is only available on 64-bit targets}}
}

unsigned long long testdivdeu(unsigned long long dividend, unsigned long long divisor) {
  return __divdeu(dividend, divisor); //expected-error {{this builtin is only available on 64-bit targets}}
}
#endif

unsigned long test_mfspr(void) {
  return __mfspr(cia); //expected-error {{argument to '__builtin_ppc_mfspr' must be a constant integer}}
}

void test_mtspr(void) {
   __mtspr(cia, ula); //expected-error {{argument to '__builtin_ppc_mtspr' must be a constant integer}}
}
