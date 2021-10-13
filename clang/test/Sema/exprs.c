// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

// PR 8876 - don't warn about trivially unreachable null derefs.  Note that
// we put this here because the reachability analysis only kicks in for
// suppressing false positives when code has no errors.
#define PR8876(err_ptr) do {\
    if (err_ptr) *(int*)(err_ptr) = 1;\
  } while (0)

#define PR8876_pos(err_ptr) do {\
    if (!err_ptr) *(int*)(err_ptr) = 1;\
  } while (0)


// Test that we don't report divide-by-zero errors in unreachable code.
// This test should be left as is, as it also tests CFG functionality.
void radar9171946() {
  if (0) {
    0 / (0 ? 1 : 0); // no-warning
  }
}

int test_pr8876() {
  PR8876(0); // no-warning
  PR8876_pos(0); // expected-warning{{indirection of non-volatile null pointer will be deleted, not trap}} expected-note{{consider using __builtin_trap() or qualifying pointer with 'volatile'}}
  return 0;
}

// PR 8183 - Handle null pointer constants on the left-side of the '&&', and reason about
// this when determining the reachability of the null pointer dereference on the right side.
void pr8183(unsigned long long test)
{
  (void)((((void*)0)) && (*((unsigned long long*)(((void*)0))) = ((unsigned long long)((test)) % (unsigned long long)((1000000000)))));  // no-warning
  (*((unsigned long long*)(((void*)0))) = ((unsigned long long)((test)) % (unsigned long long)((1000000000)))); // expected-warning {{indirection of non-volatile null pointer will be deleted, not trap}} expected-note {{consider using __builtin_trap() or qualifying pointer with 'volatile'}}
}

// PR1966
_Complex double test1() {
  return __extension__ 1.0if;
}

_Complex double test2() {
  return 1.0if;    // expected-warning {{imaginary constants are a GNU extension}}
}

// rdar://6097308
void test3() {
  int x;
  (__extension__ x) = 10;
}

// rdar://6162726
void test4() {
      static int var;
      var =+ 5;  // expected-warning {{use of unary operator that may be intended as compound assignment (+=)}}
      var =- 5;  // expected-warning {{use of unary operator that may be intended as compound assignment (-=)}}
      var = +5;  // no warning when space between the = and +.
      var = -5;

      var =+5;  // no warning when the subexpr of the unary op has no space before it.
      var =-5;
  
#define FIVE 5
      var=-FIVE;  // no warning with macros.
      var=-FIVE;
}

// rdar://6319320
void test5(int *X, float *P) {
  (float*)X = P;   // expected-error {{assignment to cast is illegal, lvalue casts are not supported}}
#define FOO ((float*) X)
  FOO = P;   // expected-error {{assignment to cast is illegal, lvalue casts are not supported}}
}

void test6() {
  int X;
  X();  // expected-error {{called object type 'int' is not a function or function pointer}}
}

void test7(int *P, _Complex float Gamma) {
   P = (P-42) + Gamma*4;  // expected-error {{invalid operands to binary expression ('int *' and '_Complex float')}}
}


// rdar://6095061
int test8(void) {
  int i;
  __builtin_choose_expr (0, 42, i) = 10;
  return i;
}


// PR3386
struct f { int x : 4;  float y[]; };
int test9(struct f *P) {
  int R;
  R = __alignof(P->x);  // expected-error {{invalid application of 'alignof' to bit-field}}
  R = __alignof(P->y);   // ok.
  R = sizeof(P->x); // expected-error {{invalid application of 'sizeof' to bit-field}}
  __extension__ ({ R = (__typeof__(P->x)) 2; }); // expected-error {{invalid application of 'typeof' to bit-field}}
  return R;
}

// PR3562
void test10(int n,...) {
  struct S {
    double          a[n];  // expected-error {{fields must have a constant size}}
  }               s;
  double x = s.a[0];  // should not get another error here.
}


#define MYMAX(A,B)    __extension__ ({ __typeof__(A) __a = (A); __typeof__(B) __b = (B); __a < __b ? __b : __a; })

struct mystruct {int A; };
void test11(struct mystruct P, float F) {
  MYMAX(P, F);  // expected-error {{invalid operands to binary expression ('typeof (P)' (aka 'struct mystruct') and 'typeof (F)' (aka 'float'))}}
}

// PR3753
int test12(const char *X) {
  return X == "foo";  // expected-warning {{comparison against a string literal is unspecified (use an explicit string comparison function instead)}}
}

int test12b(const char *X) {
  return sizeof(X == "foo"); // no-warning
}

// rdar://6719156
void test13(
            void (^P)()) { // expected-error {{blocks support disabled - compile with -fblocks}}
  P();
  P = ^(){}; // expected-error {{blocks support disabled - compile with -fblocks}}
}

void test14() {
  typedef long long __m64 __attribute__((__vector_size__(8)));
  typedef short __v4hi __attribute__((__vector_size__(8)));

  // Ok.
  __v4hi a;
  __m64 mask = (__m64)((__v4hi)a > (__v4hi)a);
}


// PR5242
typedef unsigned long *test15_t;

test15_t test15(void) {
  return (test15_t)0 + (test15_t)0;  // expected-error {{invalid operands to binary expression ('test15_t' (aka 'unsigned long *') and 'test15_t')}}
}

// rdar://7446395
void test16(float x) { x == ((void*) 0); }  // expected-error {{invalid operands to binary expression}}

// PR6004
void test17(int x) {
  x = x / 0;  // expected-warning {{division by zero is undefined}}
  x = x % 0;  // expected-warning {{remainder by zero is undefined}}
  x /= 0;  // expected-warning {{division by zero is undefined}}
  x %= 0;  // expected-warning {{remainder by zero is undefined}}
  
  x = sizeof(x/0);  // no warning.
}

// PR6501, PR11857, and PR23564
void test18_a(int a); // expected-note 2 {{'test18_a' declared here}}
void test18_b(int); // expected-note {{'test18_b' declared here}}
void test18_c(int a, int b); // expected-note 2 {{'test18_c' declared here}}
void test18_d(int a, ...); // expected-note {{'test18_d' declared here}}
void test18_e(int a, int b, ...); // expected-note {{'test18_e' declared here}}
#define MY_EXPORT __attribute__((visibility("default")))
MY_EXPORT void // (no "declared here" notes on this line, no "expanded from MY_EXPORT" notes either)
test18_f(int a, int b); // expected-note 2 {{'test18_f' declared here}}
void test18(int b) {
  test18_a(b, b); // expected-error {{too many arguments to function call, expected single argument 'a', have 2}}
  test18_a(); // expected-error {{too few arguments to function call, single argument 'a' was not specified}}
  test18_b(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  test18_c(b); // expected-error {{too few arguments to function call, expected 2, have 1}}
  test18_c(b, b, b); // expected-error {{too many arguments to function call, expected 2, have 3}}
  test18_d(); // expected-error {{too few arguments to function call, at least argument 'a' must be specified}}
  test18_e(); // expected-error {{too few arguments to function call, expected at least 2, have 0}}
  test18_f(b); // expected-error {{too few arguments to function call, expected 2, have 1}}
  test18_f(b, b, b); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

typedef int __attribute__((address_space(256))) int_AS256;
// PR7569
void test19() {
  *(int *)0 = 0;                                     // expected-warning {{indirection of non-volatile null pointer}} \
                  // expected-note {{consider using __builtin_trap}}
  *(volatile int *)0 = 0;                            // Ok.
  *(int __attribute__((address_space(256))) *)0 = 0; // Ok.
  *(int __attribute__((address_space(0))) *)0 = 0;   // expected-warning {{indirection of non-volatile null pointer}} \
                     // expected-note {{consider using __builtin_trap}}
  *(int_AS256 *)0 = 0;                               // Ok.

  // rdar://9269271
  int x = *(int *)0;                                                                          // expected-warning {{indirection of non-volatile null pointer}} \
                     // expected-note {{consider using __builtin_trap}}
  int x2 = *(volatile int *)0;                                                                // Ok.
  int x3 = *(int __attribute__((address_space(0))) *)0;                                       // expected-warning {{indirection of non-volatile null pointer}} \
                     // expected-note {{consider using __builtin_trap}}
  int x4 = *(int_AS256 *)0;                                                                   // Ok.
  int *p = &(*(int *)0);                                                                      // Ok.
  int_AS256 *p1 = &(*(int __attribute__((address_space(256))) *)0);                           // Ok.
  int __attribute__((address_space(0))) *p2 = &(*(int __attribute__((address_space(0))) *)0); // Ok.
}

int test20(int x) {
  return x && 4; // expected-warning {{use of logical '&&' with constant operand}} \
                 // expected-note {{use '&' for a bitwise operation}} \
                 // expected-note {{remove constant to silence this warning}}

  return x && sizeof(int) == 4;  // no warning, RHS is logical op.
  
  // no warning, this is an idiom for "true" in old C style.
  return x && (signed char)1;

  return x || 0;
  return x || 1;
  return x || -1; // expected-warning {{use of logical '||' with constant operand}} \
                  // expected-note {{use '|' for a bitwise operation}}
  return x || 5; // expected-warning {{use of logical '||' with constant operand}} \
                 // expected-note {{use '|' for a bitwise operation}}
  return x && 0;
  return x && 1;
  return x && -1; // expected-warning {{use of logical '&&' with constant operand}} \
                  // expected-note {{use '&' for a bitwise operation}} \
                  // expected-note {{remove constant to silence this warning}}
  return x && 5; // expected-warning {{use of logical '&&' with constant operand}} \
                 // expected-note {{use '&' for a bitwise operation}} \
                 // expected-note {{remove constant to silence this warning}}
  return x || (0);
  return x || (1);
  return x || (-1); // expected-warning {{use of logical '||' with constant operand}} \
                    // expected-note {{use '|' for a bitwise operation}}
  return x || (5); // expected-warning {{use of logical '||' with constant operand}} \
                   // expected-note {{use '|' for a bitwise operation}}
  return x && (0);
  return x && (1);
  return x && (-1); // expected-warning {{use of logical '&&' with constant operand}} \
                    // expected-note {{use '&' for a bitwise operation}} \
                    // expected-note {{remove constant to silence this warning}}
  return x && (5); // expected-warning {{use of logical '&&' with constant operand}} \
                   // expected-note {{use '&' for a bitwise operation}} \
                   // expected-note {{remove constant to silence this warning}}

}

struct Test21; // expected-note 2 {{forward declaration}}
void test21(volatile struct Test21 *ptr) {
  void test21_help(void);
  (test21_help(), *ptr); // expected-error {{incomplete type 'struct Test21' where a complete type is required}}
  (*ptr, test21_help()); // expected-error {{incomplete type 'struct Test21' where a complete type is required}}
}

// Make sure we do function/array decay.
void test22() {
  if ("help")
    (void) 0;

  if (test22) // expected-warning {{address of function 'test22' will always evaluate to 'true'}} \
	      // expected-note {{prefix with the address-of operator to silence this warning}}
    (void) 0;

  if (&test22)
    (void) 0;
}
