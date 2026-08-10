// RUN: %clang_cc1 -verify %s

typedef unsigned int uint4 __attribute((ext_vector_type(4)));
typedef int int4 __attribute((ext_vector_type(4)));
typedef int int3 __attribute((ext_vector_type(3)));
typedef unsigned uint3 __attribute((ext_vector_type(3)));

void vector_conv_invalid(const global int4 *const_global_ptr) {
  uint4 u = (uint4)(1);
  int4 i = u; // expected-error{{initializing '__private int4' (vector of 4 'int' values) with an expression of incompatible type '__private uint4' (vector of 4 'unsigned int' values)}}
  int4 e = (int4)u; // expected-error{{invalid conversion between ext-vector type 'int4' (vector of 4 'int' values) and 'uint4' (vector of 4 'unsigned int' values)}}

  uint3 u4 = (uint3)u; // expected-error{{invalid conversion between ext-vector type 'uint3' (vector of 3 'unsigned int' values) and 'uint4' (vector of 4 'unsigned int' values)}}

  e = (const int4)i;
  e = (constant int4)i;
  e = (private int4)i;

  private int4 *private_ptr = (const private int4 *)const_global_ptr; // expected-error{{casting 'const __global int4 *' to type 'const __private int4 *' changes address space of pointer}}
  global int4 *global_ptr = const_global_ptr;                 // expected-warning {{initializing '__global int4 *__private' with an expression of type 'const __global int4 *__private' discards qualifiers}}
  global_ptr = (global int4 *)const_global_ptr;
}
