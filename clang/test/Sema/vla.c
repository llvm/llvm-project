// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic

int test1(void) {
  typedef int x[test1()];  // vla
  static int y = sizeof(x);  // expected-error {{not a compile-time constant}}
}

// PR2347
void f (unsigned int m)
{
  int e[2][m];

  e[0][0] = 0;
}

// PR3048
int x = sizeof(struct{char qq[x];}); // expected-error {{fields must have a constant size}}

// PR2352
void f2(unsigned int m)
{
  extern int e1[2][m]; // expected-error {{variable length array declaration cannot have 'extern' linkage}}

  e1[0][0] = 0;

}

// PR2361
int i;
int c[][i]; // expected-error {{variably modified type declaration not allowed at file scope}}
int d[i]; // expected-error {{variable length array declaration not allowed at file scope}}

int (*e)[i]; // expected-error {{variably modified type declaration not allowed at file scope}}

void f3(void)
{
  static int a[i]; // expected-error {{variable length array declaration cannot have 'static' storage duration}}
  extern int b[i]; // expected-error {{variable length array declaration cannot have 'extern' linkage}}

  extern int (*c1)[i]; // expected-error {{variably modified type declaration cannot have 'extern' linkage}}
  static int (*d)[i];
}

// PR3663
static const unsigned array[((2 * (int)((((4) / 2) + 1.0/3.0) * (4) - 1e-8)) + 1)]; // expected-warning {{variable length array folded to constant array as an extension}}

int a[*]; // expected-error {{star modifier used outside of function prototype}}
int f4(int a[*][*]);

// PR2044
int pr2044(int b) {int (*c(void))[b];**c() = 2;} // expected-error {{variably modified type}}
int pr2044b;
int (*pr2044c(void))[pr2044b]; // expected-error {{variably modified type}}

const int f5_ci = 1;
void f5(void) { char a[][f5_ci] = {""}; } // expected-error {{variable-sized object may not be initialized}}

// PR5185
void pr5185(int a[*]);
void pr5185(int a[*]) // expected-error {{variable length array must be bound in function definition}}
{
}

void pr23151(int (*p1)[*]) // expected-error {{variable length array must be bound in function definition}}
{}

// Make sure this isn't treated as an error
int TransformBug(int a) {
 return sizeof(*(int(*)[({ goto v; v: a;})]) 0); // expected-warning {{use of GNU statement expression extension}}
}

// PR36157
struct {
  int a[ // expected-error {{variable length array in struct}}
    implicitly_declared() // expected-error {{call to undeclared function 'implicitly_declared'; ISO C99 and later do not support implicit function declarations}}
  ];
};
int (*use_implicitly_declared)(void) = implicitly_declared; // ok, was implicitly declared at file scope

void VLAPtrAssign(int size) {
  int array[1][2][3][size][4][5];
  // This is well formed
  int (*p)[2][3][size][4][5] = array;
  // Last array dimension too large
  int (*p2)[2][3][size][4][6] = array; // expected-warning {{incompatible pointer types}}
  // Second array dimension too large
  int (*p3)[20][3][size][4][5] = array; // expected-warning {{incompatible pointer types}}

  // Not illegal in C, program _might_ be well formed if size == 3.
  int (*p4)[2][size][3][4][5] = array;
}

void pr44406(void) {
  goto L; // expected-error {{cannot jump}}
  int z[(int)(1.0 * 2)]; // expected-note {{bypasses initialization of variable length array}}
L:;
}

const int pr44406_a = 32;
typedef struct {
  char c[pr44406_a]; // expected-warning {{folded to constant array as an extension}}
} pr44406_s;

void test_fold_to_constant_array(void) {
  const int ksize = 4;

  goto jump_over_a1; // expected-error{{cannot jump from this goto statement to its label}}
  char a1[ksize]; // expected-note{{variable length array}}
 jump_over_a1:;

  goto jump_over_a2;
  char a2[ksize] = "foo"; // expected-warning{{variable length array folded to constant array as an extension}}
 jump_over_a2:;

  goto jump_over_a3;
  char a3[ksize] = {}; // expected-warning {{variable length array folded to constant array as an extension}} expected-warning{{use of an empty initializer is a C23 extension}}
 jump_over_a3:;

  goto jump_over_a4; // expected-error{{cannot jump from this goto statement to its label}}
  char a4[ksize][2]; // expected-note{{variable length array}}
 jump_over_a4:;

  char a5[ksize][2] = {}; // expected-warning {{variable length array folded to constant array as an extension}} expected-warning{{use of an empty initializer is a C23 extension}}

  int a6[ksize] = {1,2,3,4}; // expected-warning{{variable length array folded to constant array as an extension}}

  // expected-warning@+1{{variable length array folded to constant array as an extension}}
  int a7[ksize] __attribute__((annotate("foo"))) = {1,2,3,4};

  // expected-warning@+1{{variable length array folded to constant array as an extension}}
  char a8[2][ksize] = {{1,2,3,4},{4,3,2,1}};

  // expected-warning@+1{{variable length array folded to constant array as an extension}}
  char (*a9)[] = (char[2][ksize]) {{1,2,3,4},{4,3,2,1}};

  char (*a10)[ksize] = 0;
}
