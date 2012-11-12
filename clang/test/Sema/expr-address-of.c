// RUN: %clang_cc1 %s -verify -fsyntax-only
struct xx { int bitf:1; };

struct entry { struct xx *whatever; 
               int value; 
               int bitf:1; };
void add_one(int *p) { (*p)++; }

void test() {
 register struct entry *p;
 add_one(&p->value);
 struct entry pvalue;
 add_one(&p->bitf);  // expected-error {{address of bit-field requested}}
 add_one(&pvalue.bitf); // expected-error {{address of bit-field requested}}
 add_one(&p->whatever->bitf); // expected-error {{address of bit-field requested}}
}

void foo() {
  register int x[10];
  &x[10];              // expected-error {{address of register variable requested}}
    
  register int *y;
  
  int *x2 = &y; // expected-error {{address of register variable requested}}
  int *x3 = &y[10];
}

void testVectorComponentAccess() {
  typedef float v4sf __attribute__ ((vector_size (16)));
  static v4sf q;
  float* r = &q[0]; // expected-error {{address of vector element requested}}
}

typedef __attribute__(( ext_vector_type(4) ))  float float4;

float *testExtVectorComponentAccess(float4 x) { 
  return &x.w; // expected-error {{address of vector element requested}}
}

void f0() {
  register int *x0;
  int *_dummy0 = &(*x0);

  register int *x1;
  int *_dummy1 = &(*(x1 + 1));
}

// FIXME: The checks for this function are broken; we should error
// on promoting a register array to a pointer! (C99 6.3.2.1p3)
void f1() {
  register int x0[10];
  int *_dummy00 = x0; // fixme-error {{address of register variable requested}}
  int *_dummy01 = &(*x0); // fixme-error {{address of register variable requested}}

  register int x1[10];
  int *_dummy1 = &(*(x1 + 1)); // fixme-error {{address of register variable requested}}

  register int *x2;
  int *_dummy2 = &(*(x2 + 1));

  register int x3[10][10][10];
  int (*_dummy3)[10] = &x3[0][0]; // expected-error {{address of register variable requested}}

  register struct { int f0[10]; } x4;
  int *_dummy4 = &x4.f0[2]; // expected-error {{address of register variable requested}}
}

void f2() {
  register int *y;
  
  int *_dummy0 = &y; // expected-error {{address of register variable requested}}
  int *_dummy1 = &y[10];
}

void f3() {
  extern void f4();
  void (*_dummy0)() = &****f4;
}

void f4() {
  register _Complex int x;
  
  int *_dummy0 = &__real__ x; // expected-error {{address of register variable requested}}
}

void f5() {
  register int arr[2];

  /* This is just here because if we happened to support this as an
     lvalue we would need to give a warning. Note that gcc warns about
     this as a register before it warns about it as an invalid
     lvalue. */
  int *_dummy0 = &(int*) arr; // expected-error {{address expression must be an lvalue or a function designator}}
  int *_dummy1 = &(arr + 1); // expected-error {{address expression must be an lvalue or a function designator}}
}

void f6(register int x) {
  int * dummy0 = &x; // expected-error {{address of register variable requested}}
}

char* f7() {
  register struct {char* x;} t1 = {"Hello"};
  char* dummy1 = &(t1.x[0]);

  struct {int a : 10;} t2;
  int* dummy2 = &(t2.a); // expected-error {{address of bit-field requested}}

  void* t3 = &(*(void*)0);
}

void f8() {
  void *dummy0 = &f8(); // expected-error {{address expression must be an lvalue or a function designator}}

  extern void v;
  void *dummy1 = &(1 ? v : f8()); // expected-error {{address expression must be an lvalue or a function designator}}

  void *dummy2 = &(f8(), v); // expected-error {{address expression must be an lvalue or a function designator}}

  void *dummy3 = &({ ; }); // expected-error {{address expression must be an lvalue or a function designator}}
}
