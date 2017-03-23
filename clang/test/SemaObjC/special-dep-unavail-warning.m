// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://8769853

@interface B
- (void) depInA;
- (void) unavailMeth __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depInA1 __attribute__((deprecated)); // expected-note {{'depInA1' has been explicitly marked deprecated here}}
- (void) unavailMeth1;
- (void) depInA2 __attribute__((deprecated)); // expected-note {{'depInA2' has been explicitly marked deprecated here}}
- (void) unavailMeth2 __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depunavailInA;
- (void) depunavailInA1 __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void)FuzzyMeth __attribute__((deprecated)); // expected-note {{'FuzzyMeth' has been explicitly marked deprecated here}}
- (void)FuzzyMeth1 __attribute__((unavailable));
@end

@interface A
- (void) unavailMeth1 __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depInA __attribute__((deprecated)); // expected-note {{'depInA' has been explicitly marked deprecated here}}
- (void) depInA2 __attribute__((deprecated));
- (void) depInA1;
- (void) unavailMeth2 __attribute__((unavailable)); 
- (void) depunavailInA __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depunavailInA1;
- (void)FuzzyMeth __attribute__((unavailable));
- (void)FuzzyMeth1 __attribute__((deprecated)); // expected-note {{'FuzzyMeth1' has been explicitly marked deprecated here}}
@end


@class C;	// expected-note 10 {{forward declaration of class here}}

void test(C *c) {
  [c depInA]; // expected-warning {{'depInA' may be deprecated because the receiver type is unknown}}
  [c unavailMeth]; // expected-warning {{'unavailMeth' may be unavailable because the receiver type is unknown}}
  [c depInA1]; // expected-warning {{'depInA1' may be deprecated because the receiver type is unknown}}
  [c unavailMeth1]; // expected-warning {{'unavailMeth1' may be unavailable because the receiver type is unknown}}
  [c depInA2]; // expected-warning {{'depInA2' may be deprecated because the receiver type is unknown}}
  [c unavailMeth2]; // expected-warning {{'unavailMeth2' may be unavailable because the receiver type is unknown}}
  [c depunavailInA]; // expected-warning {{'depunavailInA' may be unavailable because the receiver type is unknown}} 
  [c depunavailInA1];// expected-warning {{'depunavailInA1' may be unavailable because the receiver type is unknown}}
  [c FuzzyMeth];      // expected-warning {{'FuzzyMeth' may be deprecated because the receiver type is unknown}}
  [c FuzzyMeth1]; // expected-warning {{'FuzzyMeth1' may be deprecated because the receiver type is unknown}}

}

// rdar://10268422
__attribute ((deprecated))
@interface DEPRECATED // expected-note {{'DEPRECATED' has been explicitly marked deprecated here}}
+(id)new;
@end

void foo() {
  [DEPRECATED new]; // expected-warning {{'DEPRECATED' is deprecated}}
}

