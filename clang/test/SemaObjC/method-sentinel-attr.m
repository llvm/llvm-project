// RUN: %clang_cc1  -fsyntax-only -verify %s

#define NULL (void*)0

#define ATTR __attribute__ ((__sentinel__)) 

@interface INTF
- (void) foo1 : (int)x, ... ATTR; // expected-note {{method has been explicitly marked sentinel here}}
- (void) foo3 : (int)x __attribute__ ((__sentinel__)) ; // expected-warning {{'sentinel' attribute only supported for variadic functions}}
- (void) foo5 : (int)x, ... __attribute__ ((__sentinel__(1))); // expected-note {{method has been explicitly marked sentinel here}}
- (void) foo6 : (int)x, ... __attribute__ ((__sentinel__(5))); // expected-note {{method has been explicitly marked sentinel here}}
- (void) foo7 : (int)x, ... __attribute__ ((__sentinel__(0))); // expected-note {{method has been explicitly marked sentinel here}}
- (void) foo8 : (int)x, ... __attribute__ ((__sentinel__("a")));  // expected-error {{'sentinel' attribute requires parameter 1 to be an integer constant}}
- (void) foo9 : (int)x, ... __attribute__ ((__sentinel__(-1)));  // expected-error {{'sentinel' parameter 1 less than zero}}
- (void) foo10 : (int)x, ... __attribute__ ((__sentinel__(1,1)));
- (void) foo11 : (int)x, ... __attribute__ ((__sentinel__(1,1,3)));  // expected-error {{attribute takes no more than 2 arguments}}
- (void) foo12 : (int)x, ... ATTR; // expected-note {{method has been explicitly marked sentinel here}}

// rdar://7975788
- (id) foo13 : (id)firstObj, ... __attribute__((sentinel(0,1)));
- (id) foo14 : (id)firstObj :  (Class)secondObj, ... __attribute__((sentinel(0,1)));
- (id) foo15 : (id*)firstObj, ... __attribute__((sentinel(0,1)));
- (id) foo16 : (id**)firstObj, ... __attribute__((sentinel(0,1)));
@end

int main ()
{
  INTF *p;

  [p foo1:1, NULL]; // OK
  [p foo1:1, 0];  // expected-warning {{missing sentinel in method dispatch}}
  [p foo5:1, NULL, 2]; // OK
  [p foo5:1, 2, NULL, 1]; // OK
  [p foo5:1, NULL, 2, 1];  // expected-warning {{missing sentinel in method dispatch}}

  [p foo6:1,2,3,4,5,6,7];  // expected-warning {{missing sentinel in method dispatch}}
  [p foo6:1,NULL,3,4,5,6,7]; // OK
  [p foo7:1];	 // expected-warning {{not enough variable arguments in 'foo7:' declaration to fit a sentinel}}
  [p foo7:1, NULL]; // ok

  [p foo12:1]; // expected-warning {{not enough variable arguments in 'foo12:' declaration to fit a sentinel}}

  // rdar://7975788
  [ p foo13 : NULL]; 
  [ p foo14 : 0 : NULL]; 
  [ p foo16 : NULL]; 
  [ p foo15 : NULL];
}
 
