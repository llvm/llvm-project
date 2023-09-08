// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zfh \
// RUN:   -target-feature +zvfh -disable-O0-optnone %s -fsyntax-only -verify

#include <riscv_vector.h>

vuint64m2_t test_vlmul_trunc_v_u64m2_u64m2(vuint64m2_t op1) { // expected-note {{'test_vlmul_trunc_v_u64m2_u64m2' declared here}}
  return __riscv_vlmul_trunc_v_u64m2_u64m2(op1); // expected-error {{call to undeclared function '__riscv_vlmul_trunc_v_u64m2_u64m2'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vuint64m2_t' (aka '__rvv_uint64m2_t')}} expected-note {{did you mean 'test_vlmul_trunc_v_u64m2_u64m2'?}}
}

vuint64m4_t test_vlmul_trunc_v_u64m4_u64m4(vuint64m4_t op1) { // expected-note {{'test_vlmul_trunc_v_u64m4_u64m4' declared here}}
  return __riscv_vlmul_trunc_v_u64m4_u64m4(op1); // expected-error {{call to undeclared function '__riscv_vlmul_trunc_v_u64m4_u64m4'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vuint64m4_t' (aka '__rvv_uint64m4_t')}} expected-note {{did you mean 'test_vlmul_trunc_v_u64m4_u64m4'?}}
}

vuint64m1_t test_vlmul_trunc_v_u64m1_u64m1(vuint64m1_t op1) { // expected-note {{'test_vlmul_trunc_v_u64m1_u64m1' declared here}}
  return __riscv_vlmul_trunc_v_u64m1_u64m1(op1); // expected-error {{call to undeclared function '__riscv_vlmul_trunc_v_u64m1_u64m1'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vuint64m1_t' (aka '__rvv_uint64m1_t')}} expected-note {{did you mean 'test_vlmul_trunc_v_u64m1_u64m1'?}}
}

vuint64m8_t test_vlmul_trunc_v_u64m8_u64m8(vuint64m8_t op1) { // expected-note {{'test_vlmul_trunc_v_u64m8_u64m8' declared here}}
  return __riscv_vlmul_trunc_v_u64m8_u64m8(op1); // expected-error {{call to undeclared function '__riscv_vlmul_trunc_v_u64m8_u64m8'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vuint64m8_t' (aka '__rvv_uint64m8_t')}} expected-note {{did you mean 'test_vlmul_trunc_v_u64m8_u64m8'?}}
}
