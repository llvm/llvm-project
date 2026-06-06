// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1(void) {
  int x = 0;
/* expected-error@+1 {{'#pragma fp_contract' can only appear at file scope or at the start of a compound statement}} */
#pragma STDC FP_CONTRACT ON
}

void f2(void) {
  #pragma STDC FP_CONTRACT OFF
  #pragma STDC FP_CONTRACT ON 
}

struct S1 {
// expected-error@+1 {{this pragma cannot appear in struct declaration}}
#pragma STDC FP_CONTRACT ON
  float f1;
};

union U1 {
  float f1;
  float f2;
// expected-error@+1 {{this pragma cannot appear in union declaration}}
#pragma STDC FP_CONTRACT ON
};

float fp_reassoc_fail(float a, float b) {
  // CHECK-LABEL: fp_reassoc_fail
  // expected-error@+2{{'#pragma clang fp' can only appear at file scope or at the start of a compound statement}}
  float c = a + b;
#pragma clang fp reassociate(off)
  return c - b;
}

float fp_reassoc_no_fast(float a, float b) {
// CHECK-LABEL: fp_reassoc_no_fast
// expected-error@+1{{unexpected argument 'fast' to '#pragma clang fp reassociate'; expected 'on' or 'off'}}
#pragma clang fp reassociate(fast)
  return a - b;
}

float fp_recip_fail(float a, float b) {
  // CHECK-LABEL: fp_recip_fail
  // expected-error@+2{{'#pragma clang fp' can only appear at file scope or at the start of a compound statement}}
  float c = a + b;
#pragma clang fp reciprocal(off)
  return c - b;
}

float fp_recip_no_fast(float a, float b) {
// CHECK-LABEL: fp_recip_no_fast
// expected-error@+1{{unexpected argument 'fast' to '#pragma clang fp reciprocal'; expected 'on' or 'off'}}
#pragma clang fp reciprocal(fast)
  return a - b;
}
