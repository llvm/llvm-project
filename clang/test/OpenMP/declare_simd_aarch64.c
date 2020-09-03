// REQUIRES: aarch64-registered-target
// -fopemp and -fopenmp-simd behavior are expected to be the same.

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fopenmp -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s --check-prefix=AARCH64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fopenmp-simd -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s --check-prefix=AARCH64

#pragma omp declare simd
#pragma omp declare simd simdlen(2)
#pragma omp declare simd simdlen(6)
#pragma omp declare simd simdlen(8)
double foo(float x);

// AARCH64: "_ZGVnM2v_foo" "_ZGVnM4v_foo" "_ZGVnM8v_foo" "_ZGVnN2v_foo" "_ZGVnN4v_foo" "_ZGVnN8v_foo"
// AARCH64-NOT: _ZGVnN6v_foo

void foo_loop(double *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = foo(y[i]);
  }
}

// make sure that the following two function by default gets generated
// with 4 and 2 lanes, as descrived in the vector ABI
#pragma omp declare simd notinbranch
float bar(double x);
#pragma omp declare simd notinbranch
double baz(float x);

// AARCH64: "_ZGVnN2v_baz" "_ZGVnN4v_baz"
// AARCH64-NOT: baz
// AARCH64: "_ZGVnN2v_bar" "_ZGVnN4v_bar"
// AARCH64-NOT: bar

void baz_bar_loop(double *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = baz(y[i]);
    y[i] = bar(x[i]);
  }
}

  /***************************/
  /*  32-bit integer tests   */
  /***************************/

#pragma omp declare simd
#pragma omp declare simd simdlen(2)
#pragma omp declare simd simdlen(6)
#pragma omp declare simd simdlen(8)
long foo_int(int x);

// AARCH64: "_ZGVnN2v_foo_int" "_ZGVnN4v_foo_int" "_ZGVnN8v_foo_int"
// No non power of two
// AARCH64-NOT: _ZGVnN6v_foo_int

void foo_int_loop(long *x, int *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = foo_int(y[i]);
  }
}

#pragma omp declare simd
char simple_8bit(char);
// AARCH64: "_ZGVnM16v_simple_8bit" "_ZGVnM8v_simple_8bit" "_ZGVnN16v_simple_8bit" "_ZGVnN8v_simple_8bit"
#pragma omp declare simd
short simple_16bit(short);
// AARCH64: "_ZGVnM4v_simple_16bit" "_ZGVnM8v_simple_16bit" "_ZGVnN4v_simple_16bit" "_ZGVnN8v_simple_16bit"
#pragma omp declare simd
int simple_32bit(int);
// AARCH64: "_ZGVnM2v_simple_32bit" "_ZGVnM4v_simple_32bit" "_ZGVnN2v_simple_32bit" "_ZGVnN4v_simple_32bit"
#pragma omp declare simd
long simple_64bit(long);
// AARCH64: "_ZGVnM2v_simple_64bit" "_ZGVnN2v_simple_64bit"

#pragma omp declare simd
#pragma omp declare simd simdlen(32)
char a01(int x);
// AARCH64: "_ZGVnN16v_a01" "_ZGVnN32v_a01" "_ZGVnN8v_a01"
// AARCH64-NOT: a01

#pragma omp declare simd
#pragma omp declare simd simdlen(2)
long a02(short x);
// AARCH64:  "_ZGVnN2v_a02" "_ZGVnN4v_a02" "_ZGVnN8v_a02"

// AARCH64-NOT: a02
/************/
/* pointers */
/************/

#pragma omp declare simd
int b01(int *x);
// AARCH64: "_ZGVnN4v_b01"
// AARCH64-NOT: b01

#pragma omp declare simd
char b02(char *);
// AARCH64: "_ZGVnN16v_b02" "_ZGVnN8v_b02"
// AARCH64-NOT: b02

#pragma omp declare simd
double *b03(double *);
// AARCH64: "_ZGVnN2v_b03"
// AARCH64-NOT: b03

/***********/
/* masking */
/***********/

#pragma omp declare simd inbranch
int c01(double *x, short y);
// AARCH64: "_ZGVnM8vv_c01"
// AARCH64-NOT: c01

#pragma omp declare simd inbranch uniform(x)
double c02(double *x, char y);
// AARCH64: "_ZGVnM16uv_c02" "_ZGVnM8uv_c02"
// AARCH64-NOT: c02

/************************************/
/* Linear with a constant parameter */
/************************************/

#pragma omp declare simd notinbranch linear(i)
double constlinear(const int i);
// AARCH64: "_ZGVnN2l_constlinear" "_ZGVnN4l_constlinear"
// AARCH64-NOT: constlinear

/*************************/
/* sincos-like signature */
/*************************/
#pragma omp declare simd linear(sin) linear(cos)
void sincos(double in, double *sin, double *cos);
// AARCH64: "_ZGVnN2vl8l8_sincos"
// AARCH64-NOT: sincos

#pragma omp declare simd linear(sin : 1) linear(cos : 2)
void SinCos(double in, double *sin, double *cos);
// AARCH64: "_ZGVnN2vl8l16_SinCos"
// AARCH64-NOT: SinCos

// Selection of tests based on the examples provided in chapter 5 of
// the Vector Function ABI specifications for AArch64, at
// https://developer.arm.com/products/software-development-tools/hpc/arm-compiler-for-hpc/vector-function-abi.

// Listing 2, p. 18
#pragma omp declare simd inbranch uniform(x) linear(val(i) : 4)
int foo2(int *x, int i);
// AARCH64: "_ZGVnM2ul4_foo2" "_ZGVnM4ul4_foo2"
// AARCH64-NOT: foo2

// Listing 3, p. 18
#pragma omp declare simd inbranch uniform(x, c) linear(i \
                                                       : c)
int foo3(int *x, int i, unsigned char c);
// AARCH64: "_ZGVnM16uls2u_foo3" "_ZGVnM8uls2u_foo3"
// AARCH64-NOT: foo3

// Listing 6, p. 19
#pragma omp declare simd linear(x) aligned(x : 16) simdlen(4)
int foo4(int *x, float y);
// AARCH64: "_ZGVnM4l4a16v_foo4" "_ZGVnN4l4a16v_foo4"
// AARCH64-NOT: foo4

static int *I;
static char *C;
static short *S;
static long *L;
static float *F;
static double *D;
void do_something() {
  simple_8bit(*C);
  simple_16bit(*S);
  simple_32bit(*I);
  simple_64bit(*L);
  *C = a01(*I);
  *L = a02(*S);
  *I = b01(I);
  *C = b02(C);
  D = b03(D);
  *I = c01(D, *S);
  *D = c02(D, *S);
  constlinear(*I);
  sincos(*D, D, D);
  SinCos(*D, D, D);
  foo2(I, *I);
  foo3(I, *I, *C);
  foo4(I, *F);
}

typedef struct S {
  char R, G, B;
} STy;
#pragma omp declare simd notinbranch
STy DoRGB(STy x);
// AARCH64: "_ZGVnN2v_DoRGB"

static STy *RGBData;

void do_rgb_stuff() {
  DoRGB(*RGBData);
}
