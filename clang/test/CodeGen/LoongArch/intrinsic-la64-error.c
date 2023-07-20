// RUN: %clang_cc1 -triple loongarch64 -emit-llvm -S -verify %s -o /dev/null

#include <larchintrin.h>

void csrrd_d(int a) {
  __builtin_loongarch_csrrd_d(16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrrd_d(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrrd_d(a); // expected-error {{argument to '__builtin_loongarch_csrrd_d' must be a constant integer}}
}

void csrwr_d(unsigned long int a) {
  __builtin_loongarch_csrwr_d(a, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrwr_d(a, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrwr_d(a, a); // expected-error {{argument to '__builtin_loongarch_csrwr_d' must be a constant integer}}
}

void csrxchg_d(unsigned long int a, unsigned long int b) {
  __builtin_loongarch_csrxchg_d(a, b, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrxchg_d(a, b, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
  __builtin_loongarch_csrxchg_d(a, b, b); // expected-error {{argument to '__builtin_loongarch_csrxchg_d' must be a constant integer}}
}

void lddir_d(long int a, int b) {
  __builtin_loongarch_lddir_d(a, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_loongarch_lddir_d(a, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 31]}}
  __builtin_loongarch_lddir_d(a, b); // expected-error {{argument to '__builtin_loongarch_lddir_d' must be a constant integer}}
}

void ldpte_d(long int a, int b) {
  __builtin_loongarch_ldpte_d(a, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_loongarch_ldpte_d(a, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 31]}}
  __builtin_loongarch_ldpte_d(a, b); // expected-error {{argument to '__builtin_loongarch_ldpte_d' must be a constant integer}}
}

int movfcsr2gr_out_of_lo_range(int a) {
  int b = __builtin_loongarch_movfcsr2gr(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  int c = __builtin_loongarch_movfcsr2gr(32); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  int d = __builtin_loongarch_movfcsr2gr(a); // expected-error {{argument to '__builtin_loongarch_movfcsr2gr' must be a constant integer}}
  return 0;
}

void movgr2fcsr(int a, int b) {
  __builtin_loongarch_movgr2fcsr(-1, b); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(32, b); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(a, b); // expected-error {{argument to '__builtin_loongarch_movgr2fcsr' must be a constant integer}}
}
