// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s -fexperimental-new-constant-interpreter

typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));

void foo(int3 arg1, int8 arg2) {
  int4 auto1;
  int16 *auto2;
  int auto3;
  int2 auto4;
  struct S *incomplete1;

  int res1[vec_step(arg1) == 4 ? 1 : -1];
  int res2[vec_step(arg2) == 8 ? 1 : -1];
  int res3[vec_step(auto1) == 4 ? 1 : -1];
  int res4[vec_step(*auto2) == 16 ? 1 : -1];
  int res5[vec_step(auto3) == 1 ? 1 : -1];
  int res6[vec_step(auto4) == 2 ? 1 : -1];
  int res7[vec_step(int2) == 2 ? 1 : -1];
  int res8[vec_step(int3) == 4 ? 1 : -1];
  int res9[vec_step(int4) == 4 ? 1 : -1];
  int res10[vec_step(int8) == 8 ? 1 : -1];
  int res11[vec_step(int16) == 16 ? 1 : -1];
  int res12[vec_step(void) == 1 ? 1 : -1];

  int res13 = vec_step(*incomplete1); // expected-error {{'vec_step' requires built-in scalar or vector type, '__private struct S' invalid}}
  int res14 = vec_step(int16*); // expected-error {{'vec_step' requires built-in scalar or vector type, '__private int16 *' invalid}}
  int res15 = vec_step(void(void)); // expected-error {{'vec_step' requires built-in scalar or vector type, 'void (void)' invalid}}

  int res_no_effect = vec_step(auto3++); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
}
