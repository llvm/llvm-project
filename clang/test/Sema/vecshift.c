// RUN: %clang_cc1 -fsyntax-only -DERR -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-error=vec-elem-size
// RUN: %clang_cc1 -fsyntax-only -DEXT -DERR -verify %s
// RUN: %clang_cc1 -fsyntax-only -DEXT -verify %s -Wno-error=vec-elem-size

#ifdef EXT
typedef __attribute__((__ext_vector_type__(8))) char vector_char8;
typedef __attribute__((__ext_vector_type__(8))) short vector_short8;
typedef __attribute__((__ext_vector_type__(8))) int vector_int8;
typedef __attribute__((__ext_vector_type__(8))) unsigned char vector_uchar8;
typedef __attribute__((__ext_vector_type__(8))) unsigned short vector_ushort8;
typedef __attribute__((__ext_vector_type__(8))) unsigned int vector_uint8;
typedef __attribute__((__ext_vector_type__(4))) char vector_char4;
typedef __attribute__((__ext_vector_type__(4))) short vector_short4;
typedef __attribute__((__ext_vector_type__(4))) int vector_int4;
typedef __attribute__((__ext_vector_type__(4))) unsigned char vector_uchar4;
typedef __attribute__((__ext_vector_type__(4))) unsigned short vector_ushort4;
typedef __attribute__((__ext_vector_type__(4))) unsigned int vector_uint4;
#else
typedef __attribute__((vector_size(8))) char vector_char8;
typedef __attribute__((vector_size(16))) short vector_short8;
typedef __attribute__((vector_size(32))) int vector_int8;
typedef __attribute__((vector_size(8))) unsigned char vector_uchar8;
typedef __attribute__((vector_size(16))) unsigned short vector_ushort8;
typedef __attribute__((vector_size(32))) unsigned int vector_uint8;
typedef __attribute__((vector_size(4))) char vector_char4;
typedef __attribute__((vector_size(4))) short vector_short4;
typedef __attribute__((vector_size(16))) int vector_int4;
typedef __attribute__((vector_size(4))) unsigned char vector_uchar4;
typedef __attribute__((vector_size(8))) unsigned short vector_ushort4;
typedef __attribute__((vector_size(16))) unsigned int vector_uint4;
#endif

char c;
short s;
int i;
unsigned char uc;
unsigned short us;
unsigned int ui;
vector_char8 vc8;
vector_short8 vs8;
vector_int8 vi8;
vector_uchar8 vuc8;
vector_ushort8 vus8;
vector_uint8 vui8;
vector_char4 vc4;
vector_short4 vs4;
vector_int4 vi4;
vector_uchar4 vuc4;
vector_ushort4 vus4;
vector_uint4 vui4;

void foo(void) {
  vc8 = 1 << vc8;
  vuc8 = 1 << vuc8;
  vi8 = 1 << vi8;
  vui8 = 1 << vui8;
  vs8 = 1 << vs8;
  vus8 = 1 << vus8;

  vc8 = c << vc8;
  vuc8 = i << vuc8;
  vi8 = uc << vi8;
  vui8 = us << vui8;
  vs8 = ui << vs8;
  vus8 = 1 << vus8;

  vc8 = vc8 << vc8;
#ifdef ERR
  vi8 = vi8 << vuc8; // expected-error {{vector operands do not have the same elements sizes}}
  vuc8 = vuc8 << vi8; // expected-error {{vector operands do not have the same elements sizes}}
  vus8 = vus8 << vui8; // expected-error {{vector operands do not have the same elements sizes}}
  vui8 = vui8 << vs8; // expected-error {{vector operands do not have the same elements sizes}}
#else
  vi8 = vi8 << vuc8; // expected-warning {{vector operands do not have the same elements sizes}}
  vuc8 = vuc8 << vi8; // expected-warning {{vector operands do not have the same elements sizes}}
  vus8 = vus8 << vui8; // expected-warning {{vector operands do not have the same elements sizes}}
  vui8 = vui8 << vs8; // expected-warning {{vector operands do not have the same elements sizes}}
#endif

  vc8 <<= vc8;
#ifdef ERR
  vi8 <<= vuc8; // expected-error {{vector operands do not have the same elements sizes}}
  vuc8 <<= vi8; // expected-error {{vector operands do not have the same elements sizes}}
  vus8 <<= vui8; // expected-error {{vector operands do not have the same elements sizes}}
  vui8 <<= vs8; // expected-error {{vector operands do not have the same elements sizes}}
#else
  vi8 <<= vuc8; // expected-warning {{vector operands do not have the same elements sizes}}
  vuc8 <<= vi8; // expected-warning {{vector operands do not have the same elements sizes}}
  vus8 <<= vui8; // expected-warning {{vector operands do not have the same elements sizes}}
  vui8 <<= vs8; // expected-warning {{vector operands do not have the same elements sizes}}
#endif

  c <<= vc8; // expected-error {{assigning to 'char' from incompatible type}}
  i <<= vuc8; // expected-error {{assigning to 'int' from incompatible type}}
  uc <<= vi8; // expected-error {{assigning to 'unsigned char' from incompatible type}}
  us <<= vui8; // expected-error {{assigning to 'unsigned short' from incompatible type}}
  ui <<= vs8; // expected-error {{assigning to 'unsigned int' from incompatible type}}
}
