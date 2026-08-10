// RUN: %clang_cc1 -fenable-matrix -fsyntax-only %s -verify

typedef char cx4x4 __attribute__((matrix_type(4, 4)));
typedef int ix4x4 __attribute__((matrix_type(4, 4)));
typedef short sx4x4 __attribute__((matrix_type(4, 4)));
typedef int ix5x5 __attribute__((matrix_type(5, 5)));
typedef float fx5x5 __attribute__((matrix_type(5, 5)));
typedef int vec __attribute__((vector_size(4)));
typedef struct test_struct {
} test_struct;

void f1(void) {
  cx4x4 m1;
  ix4x4 m2;
  sx4x4 m3;
  ix5x5 m4;
  fx5x5 m5;
  int i;
  vec v;
  test_struct *s;

  m2 = (ix4x4)m1;
  m3 = (sx4x4)m2;
  m4 = (ix5x5)m3;        // expected-error {{conversion between matrix types 'ix5x5' (aka 'int __attribute__\
((matrix_type(5, 5)))') and 'sx4x4' (aka 'short __attribute__((matrix_type(4, 4)))') of different size \
is not allowed}}
  m5 = (ix5x5)m4;        // expected-error {{assigning to 'fx5x5' (aka \
'float __attribute__((matrix_type(5, 5)))') from incompatible type 'ix5x5' (aka 'int __attribute__((matrix_type(5, 5)))')}}
  i = (int)m4;           // expected-error {{conversion between matrix type 'ix5x5' (aka 'int __attribute__\
((matrix_type(5, 5)))') and incompatible type 'int' is not allowed}}
  m4 = (ix5x5)i;         // expected-error {{conversion between matrix type 'ix5x5' (aka 'int __attribute__\
((matrix_type(5, 5)))') and incompatible type 'int' is not allowed}}
  v = (vec)m4;           // expected-error {{conversion between matrix type 'ix5x5' (aka 'int __attribute__\
((matrix_type(5, 5)))') and incompatible type 'vec' (vector of 1 'int' value) is not allowed}}
  m4 = (ix5x5)v;         // expected-error {{conversion between matrix type 'ix5x5' (aka 'int __attribute__\
((matrix_type(5, 5)))') and incompatible type 'vec' (vector of 1 'int' value) is not allowed}}
  s = (test_struct *)m3; // expected-error {{conversion between matrix type 'sx4x4' (aka 'short \
__attribute__((matrix_type(4, 4)))') and incompatible type 'test_struct *' (aka 'struct test_struct *') is not allowed}}
  m3 = (sx4x4)s;         // expected-error {{conversion between matrix type 'sx4x4' (aka 'short \
__attribute__((matrix_type(4, 4)))') and incompatible type 'test_struct *' (aka 'struct test_struct *') is not allowed}}

  m4 = (ix5x5)m5;
}

typedef float float2_8x8 __attribute__((matrix_type(8, 8)));
typedef double double_10x10 __attribute__((matrix_type(10, 10)));
typedef double double_8x8 __attribute__((matrix_type(8, 8)));
typedef signed int signed_int_12x12 __attribute__((matrix_type(12, 12)));
typedef unsigned int unsigned_int_12x12 __attribute__((matrix_type(12, 12)));
typedef unsigned int unsigned_int_10x10 __attribute__((matrix_type(10, 10)));

void f2(void) {
  float2_8x8 m1;
  double_10x10 m2;
  double_8x8 m3;
  signed_int_12x12 m4;
  unsigned_int_12x12 m5;
  unsigned_int_10x10 m6;
  float f;

  m2 = (double_10x10)m1; // expected-error {{conversion between matrix types 'double_10x10' \
(aka 'double __attribute__((matrix_type(10, 10)))') and 'float2_8x8' (aka 'float __attribute__\
((matrix_type(8, 8)))') of different size is not allowed}}
  m3 = (double_8x8)m1;

  m5 = (unsigned_int_12x12)m4;
  m4 = (signed_int_12x12)m5;
  m6 = (unsigned_int_10x10)m4; // expected-error {{conversion between matrix types 'unsigned_int_10x10' \
(aka 'unsigned int __attribute__((matrix_type(10, 10)))') and 'signed_int_12x12' (aka 'int __attribute__\
((matrix_type(12, 12)))') of different size is not allowed}}
  f = (float)m4;               // expected-error {{conversion between matrix type 'signed_int_12x12' \
(aka 'int __attribute__((matrix_type(12, 12)))') and incompatible type 'float' is not allowed}}
  m4 = (signed_int_12x12)f;    // expected-error {{conversion between matrix type 'signed_int_12x12' \
(aka 'int __attribute__((matrix_type(12, 12)))') and incompatible type 'float' is not allowed}}
}