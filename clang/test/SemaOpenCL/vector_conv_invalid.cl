// RUN: %clang_cc1 -verify %s

typedef unsigned int uint4 __attribute((ext_vector_type(4)));
typedef int int4 __attribute((ext_vector_type(4)));
typedef int int3 __attribute((ext_vector_type(3)));
typedef unsigned uint3 __attribute((ext_vector_type(3)));

void vector_conv_invalid() {
  uint4 u = (uint4)(1);
  int4 i = u; // expected-error{{initializing 'int4' with an expression of incompatible type 'uint4'}}
  int4 e = (int4)u; // expected-error{{invalid conversion between ext-vector type 'int4' and 'uint4'}}

  uint3 u4 = (uint3)u; // expected-error{{invalid conversion between ext-vector type 'uint3' and 'uint4'}}
}
