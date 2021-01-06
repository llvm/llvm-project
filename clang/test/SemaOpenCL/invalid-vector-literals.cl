// RUN: %clang_cc1 -verify %s

typedef __attribute__(( ext_vector_type(4) ))  float float4;
typedef __attribute__(( ext_vector_type(4) ))  int int4;
typedef __attribute__(( ext_vector_type(8) ))  int int8;

void vector_literals_invalid()
{
  int4 a = (int4)(1,2,3); // expected-error{{too few elements}}
  int4 b = (int4)(1,2,3,4,5); // expected-error{{excess elements in vector}}
  ((float4)(1.0f))++; // expected-error{{cannot increment value of type 'float4'}}
  int8 d = (int8)(a,(float4)(1)); // expected-error{{initializing 'int' with an expression of incompatible type 'float4'}}
  ((int4)(0)).x = 8; // expected-error{{expression is not assignable}}
}
