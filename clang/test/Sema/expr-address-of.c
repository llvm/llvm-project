// RUN: %clang_cc1 %s -verify -fsyntax-only
struct xx { int bitf:1; };

struct entry { struct xx *whatever; 
               int value; 
               int bitf:1; };
void add_one(int *p) { (*p)++; }

void test(void) {
 register struct entry *p;
 add_one(&p->value);
 struct entry pvalue;
 add_one(&p->bitf);  // expected-error {{address of bit-field requested}}
 add_one(&pvalue.bitf); // expected-error {{address of bit-field requested}}
 add_one(&p->whatever->bitf); // expected-error {{address of bit-field requested}}
}

void foo(void) {
  register int x[10];
  &x[10];              // expected-error {{address of register variable requested}}
    
  register int *y;
  
  int *x2 = &y; // expected-error {{address of register variable requested}}
  int *x3 = &y[10];
}

void testVectorComponentAccess(void) {
  typedef float v4sf __attribute__ ((vector_size (16)));
  static v4sf q;
  float* r = &q[0]; // expected-error {{address of vector element requested}}
}

typedef __attribute__(( ext_vector_type(4) ))  float float4;

float *testExtVectorComponentAccess(float4 x) { 
  return &x.w; // expected-error {{address of vector element requested}}
}

void f0(void) {
  register int *x0;
  int *_dummy0 = &(*x0);

  register int *x1;
  int *_dummy1 = &(*(x1 + 1));
}

void f1(void) {
  register int x0[10];
  int *_dummy00 = x0;     // expected-error {{address of register variable requested}}
  int *_dummy01 = &(*x0); // expected-error {{address of register variable requested}}

  register int x1[10];
  int *_dummy1 = &(*(x1 + 1)); // expected-error {{address of register variable requested}}

  register int *x2;
  int *_dummy2 = &(*(x2 + 1));

  register int x3[10][10][10];
  int(*_dummy3)[10] = &x3[0][0]; // expected-error {{address of register variable requested}}

  register struct { int f0[10]; } x4;
  int *_dummy4 = &x4.f0[2]; // expected-error {{address of register variable requested}}

  add_one(x0);      // expected-error {{address of register variable requested}}
  (void)sizeof(x0); // OK, not an array decay.

  int *p = ((int *)x0)++; // expected-error {{address of register variable requested}}
}

void f2(void) {
  register int *y;
  
  int *_dummy0 = &y; // expected-error {{address of register variable requested}}
  int *_dummy1 = &y[10];
}

void f3(void) {
  extern void f4(void);
  void (*_dummy0)(void) = &****f4;
}

void f4(void) {
  register _Complex int x;
  
  int *_dummy0 = &__real__ x; // expected-error {{address of register variable requested}}
}

void f5(void) {
  register int arr[2];

  int *_dummy0 = &(int*) arr; // expected-error {{address of register variable requested}}
  int *_dummy1 = &(arr + 1); // expected-error {{address of register variable requested}}
}

void f6(register int x) {
  int * dummy0 = &x; // expected-error {{address of register variable requested}}
}

char* f7(void) {
  register struct {char* x;} t1 = {"Hello"};
  char* dummy1 = &(t1.x[0]);

  struct {int a : 10; struct{int b : 10;};} t2;
  int* dummy2 = &(t2.a); // expected-error {{address of bit-field requested}}
  int* dummy3 = &(t2.b); // expected-error {{address of bit-field requested}}

  void* t3 = &(*(void*)0);
}

void f8(void) {
  void *dummy0 = &f8(); // expected-error {{cannot take the address of an rvalue of type 'void'}}

  extern void v;
  void *dummy1 = &(1 ? v : f8()); // expected-error {{cannot take the address of an rvalue of type 'void'}}

  void *dummy2 = &(f8(), v); // expected-error {{cannot take the address of an rvalue of type 'void'}}

  void *dummy3 = &({ ; }); // expected-error {{cannot take the address of an rvalue of type 'void'}}
}

void f9(void) {
  extern void knr();
  void (*_dummy0)() = &****knr;
}
