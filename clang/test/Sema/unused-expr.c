// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unreachable-code -Wno-strict-prototypes

int foo(int X, int Y);

double sqrt(double X);  // implicitly const because of no -fmath-errno!

void bar(volatile int *VP, int *P, int A,
         _Complex double C, volatile _Complex double VC) {
  
  VP < P;              // expected-warning {{relational comparison result unused}}
  (void)A;
  (void)foo(1,2);      // no warning.
  
  A < foo(1, 2);       // expected-warning {{relational comparison result unused}}

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

extern void t1(void);
extern void t2(void);
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
    b < 1; // expected-warning{{relational comparison result unused}}
  else
    b < 2; // expected-warning{{relational comparison result unused}}
    
  while (1)
    b < 3; // expected-warning{{relational comparison result unused}}

  do
    b < 4; // expected-warning{{relational comparison result unused}}
  while (1);
  
  for (;;)
    b < 5; // expected-warning{{relational comparison result unused}}
    
  for (b < 1;;) {} // expected-warning{{relational comparison result unused}}
  for (;b < 1;) {}
  for (;;b < 1) {} // expected-warning{{relational comparison result unused}}
}

// rdar://7186119
int t5f(void) __attribute__((warn_unused_result));
void t5(void) {
  t5f();   // expected-warning {{ignoring return value of function declared with 'warn_unused_result' attribute}}
}


int fn1(void) __attribute__ ((warn_unused_result));
int fn2() __attribute__ ((pure));
int fn3() __attribute__ ((__const));
// rdar://6587766
int t6(void) {
  if (fn1() < 0 || fn2(2,1) < 0 || fn3(2) < 0)  // no warnings
    return -1;

  fn1();  // expected-warning {{ignoring return value of function declared with 'warn_unused_result' attribute}}
  fn2(92, 21);  // expected-warning {{ignoring return value of function declared with pure attribute}}
  fn3(42);  // expected-warning {{ignoring return value of function declared with const attribute}}
  __builtin_abs(0); // expected-warning {{ignoring return value of function declared with const attribute}}
  (void)0, fn1();  // expected-warning {{ignoring return value of function declared with 'warn_unused_result' attribute}}
  return 0;
}

int t7 __attribute__ ((warn_unused_result)); // expected-warning {{'warn_unused_result' attribute only applies to Objective-C methods, enums, structs, unions, classes, functions, function pointers, and typedefs}}

// PR4010
int (*fn4)(void) __attribute__ ((warn_unused_result));
void t8(void) {
  fn4(); // expected-warning {{ignoring return value of function declared with 'warn_unused_result' attribute}}
}

void t9(void) __attribute__((warn_unused_result)); // expected-warning {{attribute 'warn_unused_result' cannot be applied to functions without return value}}

// rdar://7410924
void *some_function(void);
void t10(void) {
  (void*) some_function(); //expected-warning {{expression result unused; should this cast be to 'void'?}}
}

void f(int i, ...) {
    __builtin_va_list ap;
    
    __builtin_va_start(ap, i);
    __builtin_va_arg(ap, int);
    __builtin_va_end(ap);
}

// PR8371
int fn5(void) __attribute__ ((__const));

// Don't warn for unused expressions in macro bodies; however, do warn for
// unused expressions in macro arguments. Macros below are reduced from code
// found in the wild.
#define NOP(a) (a)
#define M1(a, b) (long)foo((a), (b))
#define M2 (long)0;
#define M3(a) (t3(a), fn2())
#define M4(a, b) (foo((a), (b)) ? 0 : t3(a), 1)
#define M5(a, b) (foo((a), (b)), 1)
#define M6() fn1()
#define M7() fn2()
void t11(int i, int j) {
  M1(i, j);  // no warning
  NOP((long)foo(i, j)); // expected-warning {{expression result unused}}
  M2;  // no warning
  NOP((long)0); // expected-warning {{expression result unused}}
  M3(i); // no warning
  NOP((t3(i), fn2())); // expected-warning {{ignoring return value}}
  M4(i, j); // no warning
  NOP((foo(i, j) ? 0 : t3(i), 1)); // expected-warning {{expression result unused}}
  M5(i, j); // no warning
  NOP((foo(i, j), 1)); // expected-warning {{expression result unused}}
  M6(); // expected-warning {{ignoring return value}}
  M7(); // no warning
}
#undef NOP
#undef M1
#undef M2
#undef M3
#undef M4
#undef M5
#undef M6
#undef M7

#define UNREFERENCED_PARAMETER(x) (x)

void unused_parm(int a) {
  // Don't warn if the warning is introduced by a macro that's spelled
  // UNREFERENCED_PARAMETER, as that's a commonly used macro in Windows headers.
  UNREFERENCED_PARAMETER(a);
}
