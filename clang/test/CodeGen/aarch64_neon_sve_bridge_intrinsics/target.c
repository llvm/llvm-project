// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64 -target-feature +neon -verify -emit-llvm -o - %s

#include <arm_neon_sve_bridge.h>

__attribute__((target("sve")))
void target_sve(svint8_t s, int8x16_t n) {
  svset_neonq_s8(s, n);
  svget_neonq_s8(s);
  svdup_neonq_s8(n);
}

__attribute__((target("sve,bf16")))
void target_svebf16(svbfloat16_t t, bfloat16x8_t m) {
  svset_neonq_bf16(t, m);
  svget_neonq_bf16(t);
  svdup_neonq_bf16(m);
}

void base(int8x16_t n, bfloat16x8_t m) {
  // expected-error@+3 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  // expected-error@+2 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  // expected-error@+1 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  svset_neonq_s8(svundef_s8(), n);
  // expected-error@+2 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  // expected-error@+1 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  svget_neonq_s8(svundef_s8());
  // expected-error@+1 {{SVE vector type 'svint8_t' (aka '__SVInt8_t') cannot be used in a target without sve}}
  svdup_neonq_s8(n);

  // expected-error@+3 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  // expected-error@+2 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  // expected-error@+1 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  svset_neonq_bf16(svundef_bf16(), m);
  // expected-error@+2 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  // expected-error@+1 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  svget_neonq_bf16(svundef_bf16());
  // expected-error@+1 {{SVE vector type 'svbfloat16_t' (aka '__SVBfloat16_t') cannot be used in a target without sve}}
  svdup_neonq_bf16(m);
}
