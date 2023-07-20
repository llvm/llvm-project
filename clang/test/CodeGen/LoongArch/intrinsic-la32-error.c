// RUN: %clang_cc1 -triple loongarch32 -emit-llvm -S -verify %s -o /dev/null

#include <larchintrin.h>

void cacop_d(unsigned long int a) {
  __builtin_loongarch_cacop_d(1, a, 1024); // expected-error {{this builtin requires target: loongarch64}}
  __builtin_loongarch_cacop_w(-1, a, 1024); // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  __builtin_loongarch_cacop_w(32, a, 1024); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_loongarch_cacop_w(1, a, -4096); // expected-error {{argument value -4096 is outside the valid range [-2048, 2047]}}
  __builtin_loongarch_cacop_w(1, a, 4096); // expected-error {{argument value 4096 is outside the valid range [-2048, 2047]}}
}

void dbar(int a) {
  __builtin_loongarch_dbar(32768); // expected-error {{argument value 32768 is outside the valid range [0, 32767]}}
  __builtin_loongarch_dbar(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 32767]}}
  __builtin_loongarch_dbar(a); // expected-error {{argument to '__builtin_loongarch_dbar' must be a constant integer}}
}

void ibar(int a) {
  __builtin_loongarch_ibar(32769); // expected-error {{argument value 32769 is outside the valid range [0, 32767]}}
  __builtin_loongarch_ibar(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 32767]}}
  __builtin_loongarch_ibar(a); // expected-error {{argument to '__builtin_loongarch_ibar' must be a constant integer}}
}

void loongarch_break(int a) {
  __builtin_loongarch_break(32769); // expected-error {{argument value 32769 is outside the valid range [0, 32767]}}
  __builtin_loongarch_break(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 32767]}}
  __builtin_loongarch_break(a); // expected-error {{argument to '__builtin_loongarch_break' must be a constant integer}}
}

int movfcsr2gr_out_of_lo_range(int a) {
  int b =  __builtin_loongarch_movfcsr2gr(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  int c = __builtin_loongarch_movfcsr2gr(32); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  int d = __builtin_loongarch_movfcsr2gr(a); // expected-error {{argument to '__builtin_loongarch_movfcsr2gr' must be a constant integer}}
  return 0;
}

void movgr2fcsr(int a, int b) {
  __builtin_loongarch_movgr2fcsr(-1, b); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(32, b); // expected-error {{argument value 32 is outside the valid range [0, 3]}}
  __builtin_loongarch_movgr2fcsr(a, b); // expected-error {{argument to '__builtin_loongarch_movgr2fcsr' must be a constant integer}}
}

void syscall(int a) {
  __builtin_loongarch_syscall(32769); // expected-error {{argument value 32769 is outside the valid range [0, 32767]}}
  __builtin_loongarch_syscall(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 32767]}}
  __builtin_loongarch_syscall(a); // expected-error {{argument to '__builtin_loongarch_syscall' must be a constant integer}}
}

int crc_w_b_w(char a, int b) {
  return __builtin_loongarch_crc_w_b_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crc_w_h_w(short a, int b) {
  return __builtin_loongarch_crc_w_h_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crc_w_w_w(int a, int b) {
  return __builtin_loongarch_crc_w_w_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crc_w_d_w(long int a, int b) {
  return __builtin_loongarch_crc_w_d_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}
int crcc_w_b_w(char a, int b) {
  return __builtin_loongarch_crcc_w_b_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crcc_w_h_w(short a, int b) {
  return __builtin_loongarch_crcc_w_h_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crcc_w_w_w(int a, int b) {
  return __builtin_loongarch_crcc_w_w_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

int crcc_w_d_w(long int a, int b) {
  return __builtin_loongarch_crcc_w_d_w(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

unsigned long int csrrd_d() {
  return __builtin_loongarch_csrrd_d(1); // expected-error {{this builtin requires target: loongarch64}}
}

unsigned long int csrwr_d(unsigned long int a) {
  return __builtin_loongarch_csrwr_d(a, 1); // expected-error {{this builtin requires target: loongarch64}}
}

unsigned long int csrxchg_d(unsigned long int a, unsigned long int b) {
  return __builtin_loongarch_csrxchg_d(a, b, 1); // expected-error {{this builtin requires target: loongarch64}}
}

void csrrd_w(int a) {
    __builtin_loongarch_csrrd_w(16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrrd_w(-1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrrd_w(a); // expected-error {{argument to '__builtin_loongarch_csrrd_w' must be a constant integer}}
}

void csrwr_w(unsigned int a) {
    __builtin_loongarch_csrwr_w(a, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrwr_w(a, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrwr_w(a, a); // expected-error {{argument to '__builtin_loongarch_csrwr_w' must be a constant integer}}
}

void csrxchg_w(unsigned int a, unsigned int b) {
    __builtin_loongarch_csrxchg_w(a, b, 16384); // expected-error {{argument value 16384 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrxchg_w(a, b, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 16383]}}
    __builtin_loongarch_csrxchg_w(a, b, b); // expected-error {{argument to '__builtin_loongarch_csrxchg_w' must be a constant integer}}
}

unsigned long int iocsrrd_d(unsigned int a) {
  return __builtin_loongarch_iocsrrd_d(a); // expected-error {{this builtin requires target: loongarch64}}
}

void iocsrwr_d(unsigned long int a, unsigned int b) {
  __builtin_loongarch_iocsrwr_d(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

void asrtle_d(long int a, long int b) {
  __builtin_loongarch_asrtle_d(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

void asrtgt_d(long int a, long int b) {
  __builtin_loongarch_asrtgt_d(a, b); // expected-error {{this builtin requires target: loongarch64}}
}

void lddir_d(long int a, int b) {
  __builtin_loongarch_lddir_d(a, 1); // expected-error {{this builtin requires target: loongarch64}}
}

void ldpte_d(long int a, int b) {
  __builtin_loongarch_ldpte_d(a, 1); // expected-error {{this builtin requires target: loongarch64}}
}

void rdtime_d() {
  __rdtime_d(); // expected-error {{call to undeclared function '__rdtime_d'}}
}
