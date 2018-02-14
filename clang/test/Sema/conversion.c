// RUN: %clang_cc1 -fsyntax-only -verify -Wconversion \
// RUN:   -nostdsysteminc -nobuiltininc -isystem %S/Inputs \
// RUN:   -triple x86_64-apple-darwin %s -Wno-unreachable-code

#include <conversion.h>

#define BIG 0x7f7f7f7f7f7f7f7fL

void test0(char c, short s, int i, long l, long long ll) {
  c = c;
  c = s; // expected-warning {{implicit conversion loses integer precision}}
  c = i; // expected-warning {{implicit conversion loses integer precision}}
  c = l; // expected-warning {{implicit conversion loses integer precision}}
  s = c;
  s = s;
  s = i; // expected-warning {{implicit conversion loses integer precision}}
  s = l; // expected-warning {{implicit conversion loses integer precision}}
  i = c;
  i = s;
  i = i;
  i = l; // expected-warning {{implicit conversion loses integer precision}}
  l = c;
  l = s;
  l = i;
  l = l;

  c = (char) 0;
  c = (short) 0;
  c = (int) 0;
  c = (long) 0;
  s = (char) 0;
  s = (short) 0;
  s = (int) 0;
  s = (long) 0;
  i = (char) 0;
  i = (short) 0;
  i = (int) 0;
  i = (long) 0;
  l = (char) 0;
  l = (short) 0;
  l = (int) 0;
  l = (long) 0;

  c = (char) BIG;
  c = (short) BIG; // expected-warning {{implicit conversion from 'short' to 'char' changes value}}
  c = (int) BIG; // expected-warning {{implicit conversion from 'int' to 'char' changes value}}
  c = (long) BIG; // expected-warning {{implicit conversion from 'long' to 'char' changes value}}
  s = (char) BIG;
  s = (short) BIG;
  s = (int) BIG; // expected-warning {{implicit conversion from 'int' to 'short' changes value}}
  s = (long) BIG; // expected-warning {{implicit conversion from 'long' to 'short' changes value}}
  i = (char) BIG;
  i = (short) BIG;
  i = (int) BIG;
  i = (long) BIG; // expected-warning {{implicit conversion from 'long' to 'int' changes value}}
  l = (char) BIG;
  l = (short) BIG;
  l = (int) BIG;
  l = (long) BIG;
}

char test1(long long ll) {
  return (long long) ll; // expected-warning {{implicit conversion loses integer precision}}
}
char test1_a(long long ll) {
  return (long) ll; // expected-warning {{implicit conversion loses integer precision}}
}
char test1_b(long long ll) {
  return (int) ll; // expected-warning {{implicit conversion loses integer precision}}
}
char test1_c(long long ll) {
  return (short) ll; // expected-warning {{implicit conversion loses integer precision}}
}
char test1_d(long long ll) {
  return (char) ll;
}
char test1_e(long long ll) {
  return (long long) BIG; // expected-warning {{implicit conversion from 'long long' to 'char' changes value}}
}
char test1_f(long long ll) {
  return (long) BIG; // expected-warning {{implicit conversion from 'long' to 'char' changes value}}
}
char test1_g(long long ll) {
  return (int) BIG; // expected-warning {{implicit conversion from 'int' to 'char' changes value}}
}
char test1_h(long long ll) {
  return (short) BIG; // expected-warning {{implicit conversion from 'short' to 'char' changes value}}
}
char test1_i(long long ll) {
  return (char) BIG;
}

short test2(long long ll) {
  return (long long) ll; // expected-warning {{implicit conversion loses integer precision}}
}
short test2_a(long long ll) {
  return (long) ll; // expected-warning {{implicit conversion loses integer precision}}
}
short test2_b(long long ll) {
  return (int) ll; // expected-warning {{implicit conversion loses integer precision}}
}
short test2_c(long long ll) {
  return (short) ll;
}
short test2_d(long long ll) {
  return (char) ll;
}
short test2_e(long long ll) {
  return (long long) BIG;  // expected-warning {{implicit conversion from 'long long' to 'short' changes value}}
}
short test2_f(long long ll) {
  return (long) BIG;  // expected-warning {{implicit conversion from 'long' to 'short' changes value}}
}
short test2_g(long long ll) {
  return (int) BIG;  // expected-warning {{implicit conversion from 'int' to 'short' changes value}}
}
short test2_h(long long ll) {
  return (short) BIG;
}
short test2_i(long long ll) {
  return (char) BIG;
}

int test3(long long ll) {
  return (long long) ll;  // expected-warning {{implicit conversion loses integer precision}}
}
int test3_b(long long ll) {
  return (long) ll;  // expected-warning {{implicit conversion loses integer precision}}
}
int test3_c(long long ll) {
  return (int) ll;
}
int test3_d(long long ll) {
  return (short) ll;
}
int test3_e(long long ll) {
  return (char) ll;
}
int test3_f(long long ll) {
  return (long long) BIG;  // expected-warning {{implicit conversion from 'long long' to 'int' changes value}}
}
int test3_g(long long ll) {
  return (long) BIG; // expected-warning {{implicit conversion from 'long' to 'int' changes value}}
}
int test3_h(long long ll) {
  return (int) BIG;
}
int test3_i(long long ll) {
  return (short) BIG;
}
int test3_j(long long ll) {
  return (char) BIG;
}

long test4(long long ll) {
  return (long long) ll;
}
long test4_a(long long ll) {
  return (long) ll;
}
long test4_b(long long ll) {
  return (int) ll;
}
long test4_c(long long ll) {
  return (short) ll;
}
long test4_d(long long ll) {
  return (char) ll;
}
long test4_e(long long ll) {
  return (long long) BIG;
}
long test4_f(long long ll) {
  return (long) BIG;
}
long test4_g(long long ll) {
  return (int) BIG;
}
long test4_h(long long ll) {
  return (short) BIG;
}
long test4_i(long long ll) {
  return (char) BIG;
}

long long test5(long long ll) {
  return (long long) ll;
  return (long) ll;
  return (int) ll;
  return (short) ll;
  return (char) ll;
  return (long long) BIG;
  return (long) BIG;
  return (int) BIG;
  return (short) BIG;
  return (char) BIG;
}

void takes_char(char);
void takes_short(short);
void takes_int(int);
void takes_long(long);
void takes_longlong(long long);
void takes_float(float);
void takes_double(double);
void takes_longdouble(long double);

void test6(char v) {
  takes_char(v);
  takes_short(v);
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test7(short v) {
  takes_char(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_short(v);
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test8(int v) {
  takes_char(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_short(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test9(long v) {
  takes_char(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_short(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_int(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test10(long long v) {
  takes_char(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_short(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_int(v); // expected-warning {{implicit conversion loses integer precision}}
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test11(float v) {
  takes_char(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test12(double v) {
  takes_char(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_float(v); // expected-warning {{implicit conversion loses floating-point precision}}
  takes_double(v);
  takes_longdouble(v);
}

void test13(long double v) {
  takes_char(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit conversion turns floating-point number into integer}}
  takes_float(v); // expected-warning {{implicit conversion loses floating-point precision}}
  takes_double(v); // expected-warning {{implicit conversion loses floating-point precision}}
  takes_longdouble(v);
}

void test14(long l) {
  // Fine because of the boolean whitelist.
  char c;
  c = (l == 4);
  c = ((l <= 4) && (l >= 0));
  c = ((l <= 4) && (l >= 0)) || (l > 20);
}

void test15(char c) {
  c = c + 1 + c * 2;
  c = (short) c + 1 + c * 2; // expected-warning {{implicit conversion loses integer precision}}
}

// PR 5422
extern void *test16_external;
void test16(void) {
  int a = (unsigned long) &test16_external; // expected-warning {{implicit conversion loses integer precision}}
}

// PR 5938
void test17() {
  union {
    unsigned long long a : 8;
    unsigned long long b : 32;
    unsigned long long c;
  } U;

  unsigned int x;
  x = U.a;
  x = U.b;
  x = U.c; // expected-warning {{implicit conversion loses integer precision}} 
}

// PR 5939
void test18() {
  union {
    unsigned long long a : 1;
    unsigned long long b;
  } U;

  int x;
  x = (U.a ? 0 : 1);
  x = (U.b ? 0 : 1);
}

// None of these should warn.
unsigned char test19(unsigned long u64) {
  unsigned char x1 = u64 & 0xff;
  unsigned char x2 = u64 >> 56;

  unsigned char mask = 0xee;
  unsigned char x3 = u64 & mask;
  return x1 + x2 + x3;
}

// <rdar://problem/7631400>
void test_7631400(void) {
  // This should show up despite the caret being inside a macro substitution
  char s = LONG_MAX; // expected-warning {{implicit conversion from 'long' to 'char' changes value}}
}

// <rdar://problem/7676608>: assertion for compound operators with non-integral RHS
void f7676608(int);
void test_7676608(void) {
  float q = 0.7f;
  char c = 5;
  f7676608(c *= q);
}

// <rdar://problem/7904686>
void test_7904686(void) {
  const int i = -1;
  unsigned u1 = i; // expected-warning {{implicit conversion changes signedness}}  
  u1 = i; // expected-warning {{implicit conversion changes signedness}}  

  unsigned u2 = -1; // expected-warning {{implicit conversion changes signedness}}  
  u2 = -1; // expected-warning {{implicit conversion changes signedness}}  
}

// <rdar://problem/8232669>: don't warn about conversions required by
// contexts in system headers
void test_8232669(void) {
  unsigned bitset[20];
  SETBIT(bitset, 0);

  unsigned y = 50;
  SETBIT(bitset, y);

#define USER_SETBIT(set,bit) do { int i = bit; set[i/(8*sizeof(set[0]))] |= (1 << (i%(8*sizeof(set)))); } while(0)
  USER_SETBIT(bitset, 0); // expected-warning 2 {{implicit conversion changes signedness}}
}

// <rdar://problem/8559831>
enum E8559831a { E8559831a_val };
enum E8559831b { E8559831b_val };
typedef enum { E8559831c_val } E8559831c;
enum { E8559831d_val } value_d;

void test_8559831_a(enum E8559831a value);
void test_8559831(enum E8559831b value_a, E8559831c value_c) {
  test_8559831_a(value_a); // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}
  enum E8559831a a1 = value_a; // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}
  a1 = value_a; // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}

  test_8559831_a(E8559831b_val); // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}
  enum E8559831a a1a = E8559831b_val; // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}
  a1 = E8559831b_val; // expected-warning{{implicit conversion from enumeration type 'enum E8559831b' to different enumeration type 'enum E8559831a'}}
  
  test_8559831_a(value_c); // expected-warning{{implicit conversion from enumeration type 'E8559831c' to different enumeration type 'enum E8559831a'}}
  enum E8559831a a2 = value_c; // expected-warning{{implicit conversion from enumeration type 'E8559831c' to different enumeration type 'enum E8559831a'}}
  a2 = value_c; // expected-warning{{implicit conversion from enumeration type 'E8559831c' to different enumeration type 'enum E8559831a'}}
  
   test_8559831_a(value_d);
   enum E8559831a a3 = value_d;
   a3 = value_d;
}

void test26(int si, long sl) {
  si = sl % sl; // expected-warning {{implicit conversion loses integer precision: 'long' to 'int'}}
  si = sl % si;
  si = si % sl;
  si = si / sl;
  si = sl / si; // expected-warning {{implicit conversion loses integer precision: 'long' to 'int'}}
}

// rdar://16502418
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef __attribute__ ((ext_vector_type(16),__aligned__(32))) uint16_t ushort16;
typedef __attribute__ ((ext_vector_type( 8),__aligned__( 32))) uint32_t uint8;

void test27(ushort16 constants) {
    uint8 pairedConstants = (uint8) constants;
    ushort16 crCbScale = pairedConstants.s4; // expected-warning {{implicit conversion loses integer precision: 'uint32_t' (aka 'unsigned int') to 'ushort16'}}
    ushort16 brBias = pairedConstants.s6; // expected-warning {{implicit conversion loses integer precision: 'uint32_t' (aka 'unsigned int') to 'ushort16'}}
}


float double2float_test1(double a) {
    return a; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
}

void double2float_test2(double a, float *b) {
    *b += a;
}

float sinf (float x);
double double2float_test3(double a) {
    return sinf(a); // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
}
