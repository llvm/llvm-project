// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic

int test1() {
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
  extern int e1[2][m]; // expected-error {{variable length array declaration can not have 'extern' linkage}}

  e1[0][0] = 0;
  
}

// PR2361
int i; 
int c[][i]; // expected-error {{variably modified type declaration not allowed at file scope}}
int d[i]; // expected-error {{variable length array declaration not allowed at file scope}}

int (*e)[i]; // expected-error {{variably modified type declaration not allowed at file scope}}

void f3()
{
  static int a[i]; // expected-error {{variable length array declaration can not have 'static' storage duration}}
  extern int b[i]; // expected-error {{variable length array declaration can not have 'extern' linkage}}

  extern int (*c1)[i]; // expected-error {{variably modified type declaration can not have 'extern' linkage}}
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
void f5() { char a[][f5_ci] = {""}; } // expected-warning {{variable length array folded to constant array as an extension}}

// PR5185
void pr5185(int a[*]);
void pr5185(int a[*]) // expected-error {{variable length array must be bound in function definition}}
{
}

// Make sure this isn't treated as an error
int TransformBug(int a) {
 return sizeof(*(int(*)[({ goto v; v: a;})]) 0); // expected-warning {{use of GNU statement expression extension}}
}
