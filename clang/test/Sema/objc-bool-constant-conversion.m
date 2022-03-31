// RUN: %clang_cc1 %s -verify -fsyntax-only

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

BOOL B;

int main(void) {
  B = 0;
  B = 1;
  B = YES;
  B = NO;

  B = -1; // expected-warning{{implicit conversion from constant value -1 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
  B = 0 - 1; // expected-warning{{implicit conversion from constant value -1 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
  B = YES + YES; // expected-warning {{implicit conversion from constant value 2 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
  B = YES | YES;

  B = B ? 2 : 2; // expected-warning 2 {{implicit conversion from constant value 2 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}

  BOOL Init = -1; // expected-warning{{implicit conversion from constant value -1 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
  BOOL Init2 = B ? 2 : 2; // expected-warning 2 {{implicit conversion from constant value 2 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}

  void takesbool(BOOL);
  takesbool(43); // expected-warning {{implicit conversion from constant value 43 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}

  BOOL OutOfRange = 400; // expected-warning{{implicit conversion from constant value 400 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
}

@interface BoolProp
@property BOOL b;
@end

void f(BoolProp *bp) {
  bp.b = 43; // expected-warning {{implicit conversion from constant value 43 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
  [bp setB:43]; // expected-warning {{implicit conversion from constant value 43 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
}
