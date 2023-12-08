// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -fobjc-runtime-has-weak -I %S/Inputs -verify %s

#include "non-trivial-c-union.h"

typedef union { // expected-note 12 {{'U0' has subobjects that are non-trivial to default-initialize}} expected-note 36 {{'U0' has subobjects that are non-trivial to destruct}} expected-note 28 {{'U0' has subobjects that are non-trivial to copy}}
  id f0; // expected-note 12 {{f0 has type '__strong id' that is non-trivial to default-initialize}} expected-note 36 {{f0 has type '__strong id' that is non-trivial to destruct}} expected-note 28 {{f0 has type '__strong id' that is non-trivial to copy}}
  __weak id f1; // expected-note 12 {{f1 has type '__weak id' that is non-trivial to default-initialize}} expected-note 36 {{f1 has type '__weak id' that is non-trivial to destruct}} expected-note 28 {{f1 has type '__weak id' that is non-trivial to copy}}
} U0;

typedef struct {
  U0 f0;
  id f1;
} S0;

id g0;
U0 ug0; // expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
U0 ug1 = { .f0 = 0 };
S0 sg0; // expected-error {{cannot default-initialize an object of type 'S0' since it contains a union that is non-trivial to default-initialize}}
S0 sg1 = { .f0 = {0}, .f1 = 0 };
S0 sg2 = { .f1 = 0 }; // expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}

U0 foo0(U0); // expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to copy}}
S0 foo1(S0); // expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to copy}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to copy}}

@interface C
-(U0)m0:(U0)arg; // expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to copy}}
-(S0)m1:(S0)arg; // expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to copy}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to copy}}
@end

void testBlockFunction(void) {
  (void)^(U0 a){ return ug0; }; // expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for a function/method parameter since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to copy}}
  (void)^(S0 a){ return sg0; }; // expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for a function/method parameter since it contains a union that is non-trivial to copy}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to destruct}} expected-error {{cannot use type 'S0' for function/method return since it contains a union that is non-trivial to copy}}
}
void testAutoVar(void) {
  U0 u0; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
  U0 u1 = ug0; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot copy-initialize an object of type 'U0' since it is a union that is non-trivial to copy}}
  U0 u2 = { g0 }; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}}
  U0 u3 = { .f1 = g0 }; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}}
  S0 s0; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'S0' since it contains a union that is non-trivial to default-initialize}}
  S0 s1 = sg0; // expected-error {{declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot copy-initialize an object of type 'S0' since it contains a union that is non-trivial to copy}}
  S0 s2 = { ug0 }; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot copy-initialize an object of type 'U0' since it is a union that is non-trivial to copy}}
  S0 s3 = { .f0 = ug0 }; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot copy-initialize an object of type 'U0' since it is a union that is non-trivial to copy}}
  S0 s4 = { .f1 = g0 }; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
}

void testAssignment(void) {
  ug0 = ug1; // expected-error {{cannot assign to a variable of type 'U0' since it is a union that is non-trivial to copy}}
  sg0 = sg1; // expected-error {{cannot assign to a variable of type 'S0' since it contains a union that is non-trivial to copy}}
}

U0 ug2 = (U0){ .f1 = 0 }; // expected-error {{cannot copy-initialize an object of type 'U0' since it is a union that is non-trivial to copy}}
S0 sg3 = (S0){ .f0 = {0}, .f1 = 0 }; // expected-error {{cannot copy-initialize an object of type 'S0' since it contains a union that is non-trivial to copy}}
S0 *sg4 = &(S0){ .f1 = 0 }; // expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}

void testCompoundLiteral(void) {
  const U0 *t0 = &(U0){ .f0 = g0 }; // expected-error {{cannot construct an automatic compound literal of type 'U0' since it is a union that is non-trivial to destruct}}
  const U0 *t1 = &(U0){ .f1 = g0 }; // expected-error {{cannot construct an automatic compound literal of type 'U0' since it is a union that is non-trivial to destruct}}
  const S0 *t2 = &(S0){ .f0 = ug0 }; // expected-error {{cannot construct an automatic compound literal of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot copy-initialize an object of type 'U0' since it is a union that is non-trivial to copy}}
  const S0 *t3 = &(S0){ .f1 = g0 }; // expected-error {{cannot construct an automatic compound literal of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
}

typedef void (^BlockTy)(void);
void escapingFunc(BlockTy);
void noescapingFunc(__attribute__((noescape)) BlockTy);

void testBlockCapture(void) {
  U0 t0; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
  S0 t1; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'S0' since it contains a union that is non-trivial to default-initialize}}
  __block U0 t2; // expected-error {{cannot declare an automatic variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}
  __block S0 t3; // expected-error {{cannot declare an automatic variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot default-initialize an object of type 'S0' since it contains a union that is non-trivial to default-initialize}}

  escapingFunc(^{ g0 = t0.f0; }); // expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to copy}}
  escapingFunc(^{ g0 = t1.f0.f0; }); // expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to copy}}
  escapingFunc(^{ g0 = t2.f0; }); // expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to copy}}
  escapingFunc(^{ g0 = t3.f0.f0; }); // expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to copy}}
  noescapingFunc(^{ g0 = t0.f0; }); // expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'U0' since it is a union that is non-trivial to copy}}
  noescapingFunc(^{ g0 = t1.f0.f0; }); // expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to destruct}} expected-error {{cannot capture a variable of type 'S0' since it contains a union that is non-trivial to copy}}
  noescapingFunc(^{ g0 = t2.f0; });
  noescapingFunc(^{ g0 = t3.f0.f0; });
}

void testVolatileLValueToRValue(volatile U0 *a) {
  (void)*a; // expected-error {{cannot use volatile type 'volatile U0' where it causes an lvalue-to-rvalue conversion since it is a union that is non-trivial to destruct}} // expected-error {{cannot use volatile type 'volatile U0' where it causes an lvalue-to-rvalue conversion since it is a union that is non-trivial to copy}}
}

void unionInSystemHeader0(U0_SystemHeader);

void unionInSystemHeader1(U1_SystemHeader); // expected-error {{cannot use type 'U1_SystemHeader' for a function/method parameter since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U1_SystemHeader' for a function/method parameter since it is a union that is non-trivial to copy}}
