// RUN: %clang_cc1 %s -verify -Wobjc-signed-char-bool
// RUN: %clang_cc1 -xobjective-c++ %s -verify -Wobjc-signed-char-bool

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

typedef unsigned char Boolean;

BOOL b;
Boolean boolean;
float fl;
int i;
int *ptr;

void t1() {
  b = boolean;
  b = fl; // expected-warning {{implicit conversion from floating-point type 'float' to 'BOOL'}}
  b = i; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}

  b = 1.0;
  b = 0.0;
  b = 1.1; // expected-warning {{implicit conversion from 'double' to 'BOOL' (aka 'signed char') changes value from 1.1 to 1}}
  b = 2.1; // expected-warning {{implicit conversion from constant value 2.1 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}

  b = YES;
#ifndef __cplusplus
  b = ptr; // expected-warning {{incompatible pointer to integer conversion assigning to 'BOOL' (aka 'signed char') from 'int *'}}
#endif
}

@interface BoolProp
@property BOOL p;
@end

void t2(BoolProp *bp) {
  bp.p = YES;
  bp.p = NO;
  bp.p = boolean;
  bp.p = fl; // expected-warning {{implicit conversion from floating-point type 'float' to 'BOOL'}}
  bp.p = i; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
  bp.p = b;
  bp.p = bp.p;
#ifndef __cplusplus
  bp.p = ptr; // expected-warning {{incompatible pointer to integer conversion assigning to 'BOOL' (aka 'signed char') from 'int *'}}
#endif
  bp.p = 1;
  bp.p = 2; // expected-warning {{implicit conversion from constant value 2 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
}
