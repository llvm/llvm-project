// RUN: %clang_cc1 %s -ffreestanding -Wno-int-to-pointer-cast -fsyntax-only -verify -pedantic -fpascal-strings -std=c99

#include <stdint.h>
#include <limits.h>

int a(void) {int p; *(1 ? &p : (void*)(0 && (a(),1))) = 10;} // expected-error {{incomplete type 'void' is not assignable}}

// rdar://6091492 - ?: with __builtin_constant_p as the operand is an i-c-e.
int expr;
char w[__builtin_constant_p(expr) ? expr : 1];

char v[sizeof(__builtin_constant_p(0)) == sizeof(int) ? 1 : -1];

int implicitConversion = 1.0;
char floatArith[(int)(1.0+2.0)]; // expected-warning {{variable length array folded to constant array as an extension}}

// __builtin_constant_p as the condition of ?: allows arbitrary foldable
// constants to be transmogrified into i-c-e's.
char b[__builtin_constant_p((int)(1.0+2.0)) ? (int)(1.0+2.0) : -1];

struct c {
  int a : (
           __builtin_constant_p((int)(1.0+2.0)) ? (int)(1.0+
     expr // expected-error {{expression is not an integer constant expression}}
           ) : -1);
};

// Check that we can evaluate statement-expressions properly when
// constant-folding inside an ICE.
void PR49239(void) {
  goto check_not_vla;
  char not_vla[__builtin_constant_p(1) ? ({ 42; }) : -1]; // expected-warning {{statement expression}}
check_not_vla:;
  _Static_assert(sizeof(not_vla) == 42, ""); // expected-warning {{C11 extension}}

  // It's not clear that this should be valid: __builtin_expect(expr1, expr2)
  // should probably be an ICE if and only if expr1 is an ICE, but we roughly
  // follow GCC in treating it as an ICE if and only if we can evaluate expr1
  // regardless of whether it's an ICE.
  goto check_also_not_vla;
  char also_not_vla[__builtin_expect(({ 76; }), 0)]; // expected-warning {{statement expression}}
check_also_not_vla:;
  _Static_assert(sizeof(also_not_vla) == 76, ""); // expected-warning {{C11 extension}}
}


void test1(int n, int* p) { *(n ? p : (void *)(7-7)) = 1; }
void test2(int n, int* p) { *(n ? p : (void *)0) = 1; }



char array[1024/(sizeof (long))];

int x['\xBb' == (char) 187 ? 1: -1];

// PR1992
void func(int x)
{
  switch (x) {
    case sizeof("abc"): break;
    case sizeof("loooong"): func(4);
    case sizeof("\ploooong"): func(4);
  }
}


// rdar://4213768
int expr;
char y[__builtin_constant_p(expr) ? -1 : 1];
char z[__builtin_constant_p(4) ? 1 : -1];

// Comma tests
int comma1[0?1,2:3]; // expected-warning {{left operand of comma operator has no effect}}
int comma2[1 || (1, 2)]; // expected-warning {{use of logical '||' with constant operand}} \
                      // expected-note {{use '|' for a bitwise operation}} \
                      // expected-warning {{left operand of comma operator has no effect}}
int comma3[(1, 2)];   // expected-warning {{variable length array folded to constant array as an extension}} \
                      // expected-warning {{left operand of comma operator has no effect}}

// Pointer + __builtin_constant_p
char pbcp[__builtin_constant_p(4) ? (intptr_t)&expr : 0]; // expected-error {{variable length array declaration not allowed at file scope}}

int illegaldiv1a[1 || 1/0];
int illegaldiv1b[1 && 1/0];  //expected-error{{variable length array declaration not allowed at file scope}}

int illegaldiv2[1/0]; // expected-error {{variable length array declaration not allowed at file scope}}
int illegaldiv3[INT_MIN / -1]; // expected-error {{variable length array declaration not allowed at file scope}}
// PR9262
int illegaldiv4[0 / (1 / 0)]; // expected-error {{variable length array declaration not allowed at file scope}}

int chooseexpr[__builtin_choose_expr(1, 1, expr)];
int realop[(__real__ 4) == 4 ? 1 : -1];
int imagop[(__imag__ 4) == 0 ? 1 : -1];

int *PR14729 = 0 ?: 1/0; // expected-error {{not a compile-time constant}} expected-warning 3{{}}

int bcp_call_v;
int bcp_call_a[] = {__builtin_constant_p(bcp_call_v && 0) ? bcp_call_v && 0 : -1};
