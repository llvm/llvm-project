// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef __fp16 half4 __attribute__ ((vector_size (8)));
typedef float float4 __attribute__ ((vector_size (16)));
typedef short short4 __attribute__ ((vector_size (8)));
typedef int int4 __attribute__ ((vector_size (16)));
typedef __fp16 float16_t;

half4 hv0, hv1;
float4 fv0, fv1;
short4 sv0;
int4 iv0;

void testFP16Vec(int c) {
  hv0 = hv0 + hv1;
  hv0 = hv0 - hv1;
  hv0 = hv0 * hv1;
  hv0 = hv0 / hv1;
  hv0 = c ? hv0 : hv1;
  hv0 += hv1;
  hv0 -= hv1;
  hv0 *= hv1;
  hv0 /= hv1;
  sv0 = hv0 == hv1;
  sv0 = hv0 != hv1;
  sv0 = hv0 < hv1;
  sv0 = hv0 > hv1;
  sv0 = hv0 <= hv1;
  sv0 = hv0 >= hv1;
  sv0 = hv0 || hv1; // expected-error{{logical expression with vector types 'half4' (vector of 4 '__fp16' values) and 'half4' is only supported in C++}}
  sv0 = hv0 && hv1; // expected-error{{logical expression with vector types 'half4' (vector of 4 '__fp16' values) and 'half4' is only supported in C++}}

  // Implicit conversion between half vectors and float vectors are not allowed.
  hv0 = fv0; // expected-error{{assigning to}}
  fv0 = hv0; // expected-error{{assigning to}}
  hv0 = (half4)fv0; // expected-error{{invalid conversion between}}
  fv0 = (float4)hv0; // expected-error{{invalid conversion between}}
  hv0 = fv0 + fv1; // expected-error{{assigning to}}
  fv0 = hv0 + hv1; // expected-error{{assigning to}}
  hv0 = hv0 + fv1; // expected-error{{cannot convert between vector}}
  hv0 = c ? hv0 : fv1; // expected-error{{cannot convert between vector}}
  sv0 = hv0 == fv1; // expected-error{{cannot convert between vector}}
  sv0 = hv0 < fv1; // expected-error{{cannot convert between vector}}
  sv0 = hv0 || fv1; // expected-error{{cannot convert between vector}} expected-error{{invalid operands to binary expression}}
  iv0 = hv0 == hv1; // expected-error{{assigning to}}

  // FIXME: clang currently disallows using these operators on vectors, which is
  // allowed by gcc.
  sv0 = !hv0; // expected-error{{invalid argument type}}
  hv0++; // expected-error{{cannot increment value of type}}
  ++hv0; // expected-error{{cannot increment value of type}}
}

void testTypeDef() {
  __fp16 t0 __attribute__((vector_size (8)));
  float16_t t1 __attribute__((vector_size (8)));
  t1 = t0;
}
