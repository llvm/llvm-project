// RUN: %clang_cc1 %s -pedantic -verify -triple=x86_64-apple-darwin9
// RUN: %clang_cc1 %s -pedantic -verify -triple=mips64-linux-gnu

// rdar://6097662
typedef int (*T)[2];
restrict T x;

typedef int *S[2];
restrict S y; // expected-error {{restrict requires a pointer or reference ('S' (aka 'int *[2]') is invalid)}}



// int128_t is available.
int a() {
  __int128_t s;
  __uint128_t t;
}
// but not a keyword
int b() {
  int __int128_t;
  int __uint128_t;
}
// __int128 is a keyword
int c() {
  __int128 i;
  unsigned __int128 j;
  long unsigned __int128 k; // expected-error {{'long __int128' is invalid}}
  int __int128; // expected-error {{cannot combine with previous}} expected-warning {{does not declare anything}}
}
// __int128_t is __int128; __uint128_t is unsigned __int128.
typedef __int128 check_int_128; // expected-note {{here}}
typedef __int128_t check_int_128; // expected-note {{here}} expected-warning {{redefinition}}
typedef int check_int_128; // expected-error {{different types ('int' vs '__int128_t' (aka '__int128'))}}

typedef unsigned __int128 check_uint_128; // expected-note {{here}}
typedef __uint128_t check_uint_128; // expected-note {{here}} expected-warning {{redefinition}}
typedef int check_uint_128; // expected-error {{different types ('int' vs '__uint128_t' (aka 'unsigned __int128'))}}

// Array type merging should convert array size to whatever matches the target
// pointer size.
// rdar://6880874
extern int i[1LL];
int i[(short)1];

enum e { e_1 };
extern int j[sizeof(enum e)];  // expected-note {{previous definition}}
int j[42];   // expected-error {{redefinition of 'j' with a different type: 'int [42]' vs 'int [4]'}}

// rdar://6880104
_Decimal32 x;  // expected-error {{GNU decimal type extension not supported}}


// rdar://6880951
int __attribute__ ((vector_size (8), vector_size (8))) v;  // expected-error {{invalid vector element type}}

void test(int i) {
  char c = (char __attribute__((aligned(8)))) i; // expected-warning {{'aligned' attribute ignored when parsing type}}
}

// http://llvm.org/PR11082
//
// FIXME: This may or may not be the correct approach (no warning or error),
// but large amounts of Linux and FreeBSD code need this attribute to not be
// a hard error in order to work correctly.
void test2(int i) {
  char c = (char __attribute__((may_alias))) i;
}

// vector size too large
int __attribute__ ((vector_size(8192))) x1; // expected-error {{vector size too large}}
typedef int __attribute__ ((ext_vector_type(8192))) x2; // expected-error {{vector size too large}}

// no support for vector enum type
enum { e_2 } x3 __attribute__((vector_size(64))); // expected-error {{invalid vector element type}}

int x4 __attribute__((ext_vector_type(64)));  // expected-error {{'ext_vector_type' attribute only applies to types}}

// rdar://16492792
typedef __attribute__ ((ext_vector_type(32),__aligned__(32))) unsigned char uchar32;

void convert() {
    uchar32 r = 0;
    r.s[ 1234 ] = 1; // expected-error {{illegal vector component name 's'}}
}
