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

class C1 {
  float f1;
// expected-error@+1 {{this pragma cannot appear in class declaration}}
#pragma STDC FP_CONTRACT ON
  float f2;
};
