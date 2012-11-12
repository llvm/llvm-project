// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unreachable-code

int foo(int X, int Y);

double sqrt(double X);  // implicitly const because of no -fmath-errno!

void bar(volatile int *VP, int *P, int A,
         _Complex double C, volatile _Complex double VC) {
  
  VP < P;              // expected-warning {{expression result unused}}
  (void)A;
  (void)foo(1,2);      // no warning.
  
  A < foo(1, 2);       // expected-warning {{expression result unused}}

  foo(1,2)+foo(4,3);   // expected-warning {{expression result unused}}


  *P;                  // expected-warning {{expression result unused}}
  *VP;                 // no warning.
  P[4];                // expected-warning {{expression result unused}}
  VP[4];               // no warning.

  __real__ C;          // expected-warning {{expression result unused}}
  __real__ VC;
  
  // We know this can't change errno because of no -fmath-errno.
  sqrt(A);  // expected-warning {{ignoring return value of function declared with const attribute}}
}

extern void t1();
extern void t2();
void t3(int c) {
  c ? t1() : t2();
}

// This shouldn't warn: the expr at the end of the stmtexpr really is used.
int stmt_expr(int x, int y) {
  return ({int _a = x, _b = y; _a > _b ? _a : _b; });
}

void nowarn(unsigned char* a, unsigned char* b)
{
  unsigned char c = 1;
  *a |= c, *b += c;


  // PR4633
  int y, x;
  ((void)0), y = x;
}

void t4(int a) {
  int b = 0;

  if (a)
    b < 1; // expected-warning{{expression result unused}}
  else
    b < 2; // expected-warning{{expression result unused}}
    
  while (1)
    b < 3; // expected-warning{{expression result unused}}

  do
    b < 4; // expected-warning{{expression result unused}}
  while (1);
  
  for (;;)
    b < 5; // expected-warning{{expression result unused}}
    
  for (b < 1;;) {} // expected-warning{{expression result unused}}
  for (;b < 1;) {}
  for (;;b < 1) {} // expected-warning{{expression result unused}}
}

// rdar://7186119
int t5f(void) __attribute__((warn_unused_result));
void t5() {
  t5f();   // expected-warning {{ignoring return value of function declared with warn_unused_result}}
}


int fn1() __attribute__ ((warn_unused_result));
int fn2() __attribute__ ((pure));
int fn3() __attribute__ ((__const));
// rdar://6587766
int t6() {
  if (fn1() < 0 || fn2(2,1) < 0 || fn3(2) < 0)  // no warnings
    return -1;

  fn1();  // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
  fn2(92, 21);  // expected-warning {{ignoring return value of function declared with pure attribute}}
  fn3(42);  // expected-warning {{ignoring return value of function declared with const attribute}}
  __builtin_fabsf(0); // expected-warning {{ignoring return value of function declared with const attribute}}
  (void)0, fn1();  // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
  return 0;
}

int t7 __attribute__ ((warn_unused_result)); // expected-warning {{'warn_unused_result' attribute only applies to functions}}

// PR4010
int (*fn4)(void) __attribute__ ((warn_unused_result));
void t8() {
  fn4(); // expected-warning {{ignoring return value of function declared with warn_unused_result attribute}}
}

void t9() __attribute__((warn_unused_result)); // expected-warning {{attribute 'warn_unused_result' cannot be applied to functions without return value}}

// rdar://7410924
void *some_function(void);
void t10() {
  (void*) some_function(); //expected-warning {{expression result unused; should this cast be to 'void'?}}
}

void f(int i, ...) {
    __builtin_va_list ap;
    
    __builtin_va_start(ap, i);
    __builtin_va_arg(ap, int);
    __builtin_va_end(ap);
}

// PR8371
int fn5() __attribute__ ((__const));

// OpenSSL has some macros like this; we shouldn't warn on the cast.
#define M1(a, b) (long)foo((a), (b))
// But, we should still warn on other subexpressions of casts in macros.
#define M2 (long)0;
void t11(int i, int j) {
  M1(i, j);  // no warning
  M2;  // expected-warning {{expression result unused}}
}
#undef M1
#undef M2
