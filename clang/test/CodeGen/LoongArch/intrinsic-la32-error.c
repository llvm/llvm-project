// RUN: %clang_cc1 -triple loongarch32 -emit-llvm -verify %s -o /dev/null
// RUN: not %clang_cc1 -triple loongarch32 -DFEATURE_CHECK -emit-llvm %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

#include <larchintrin.h>

#ifdef FEATURE_CHECK
void test_feature(long *v_l, unsigned long *v_ul, int *v_i, unsigned ui, char c, short s) {
// CHECK: error: '__builtin_loongarch_cacop_d' needs target feature 64bit
  __builtin_loongarch_cacop_d(1, v_ul[0], 1024);

// CHECK: error: '__builtin_loongarch_crc_w_b_w' needs target feature 64bit
  v_i[0] = __builtin_loongarch_crc_w_b_w(c, v_i[0]);
// CHECK: error: '__builtin_loongarch_crc_w_h_w' needs target feature 64bit
  v_i[1] =  __builtin_loongarch_crc_w_h_w(c, v_i[0]);
// CHECK: error: '__builtin_loongarch_crc_w_w_w' needs target feature 64bit
  v_i[2] = __builtin_loongarch_crc_w_w_w(c, v_i[0]);
// CHECK: error: '__builtin_loongarch_crc_w_d_w' needs target feature 64bit
  v_i[3] = __builtin_loongarch_crc_w_d_w(c, v_i[0]);

// CHECK: error: '__builtin_loongarch_crcc_w_b_w' needs target feature 64bit
  v_i[4] = __builtin_loongarch_crcc_w_b_w(c, v_i[0]);
// CHECK: error: '__builtin_loongarch_crcc_w_h_w' needs target feature 64bit
  v_i[5] = __builtin_loongarch_crcc_w_h_w(s, v_i[0]);
// CHECK: error: '__builtin_loongarch_crcc_w_w_w' needs target feature 64bit
  v_i[6] = __builtin_loongarch_crcc_w_w_w(v_i[0], v_i[1]);
// CHECK: error: '__builtin_loongarch_crcc_w_d_w' needs target feature 64bit
  v_i[7] = __builtin_loongarch_crcc_w_d_w(v_l[0], v_i[0]);

// CHECK: error: '__builtin_loongarch_csrrd_d' needs target feature 64bit
  v_ul[0] = __builtin_loongarch_csrrd_d(1);
// CHECK: error: '__builtin_loongarch_csrwr_d' needs target feature 64bit
  v_ul[1] = __builtin_loongarch_csrwr_d(v_ul[0], 1);
// CHECK: error: '__builtin_loongarch_csrxchg_d' needs target feature 64bit
  v_ul[2] = __builtin_loongarch_csrxchg_d(v_ul[0], v_ul[1], 1);


// CHECK: error: '__builtin_loongarch_iocsrrd_d' needs target feature 64bit
  v_ul[3] = __builtin_loongarch_iocsrrd_d(ui);
// CHECK: error: '__builtin_loongarch_iocsrwr_d' needs target feature 64bit
  __builtin_loongarch_iocsrwr_d(v_ul[0], ui);

// CHECK: error: '__builtin_loongarch_asrtle_d' needs target feature 64bit
  __builtin_loongarch_asrtle_d(v_l[0], v_l[1]);
// CHECK: error: '__builtin_loongarch_asrtgt_d' needs target feature 64bit
  __builtin_loongarch_asrtgt_d(v_l[0], v_l[1]);

// CHECK: error: '__builtin_loongarch_lddir_d' needs target feature 64bit
  v_ul[4] = __builtin_loongarch_lddir_d(v_l[0], 1);
// CHECK: error: '__builtin_loongarch_ldpte_d' needs target feature 64bit
  __builtin_loongarch_ldpte_d(v_l[0], 1);
}
#endif

void cacop_d(unsigned long int a) {
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

void rdtime_d() {
  __rdtime_d(); // expected-error {{call to undeclared function '__rdtime_d'}}
}
