// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://8769853

@interface B
- (void) depInA;
- (void) unavailMeth __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depInA1 __attribute__((deprecated));
- (void) unavailMeth1;
- (void) depInA2 __attribute__((deprecated));
- (void) unavailMeth2 __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depunavailInA;
- (void) depunavailInA1 __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void)FuzzyMeth __attribute__((deprecated));
- (void)FuzzyMeth1 __attribute__((unavailable));
@end

@interface A
- (void) unavailMeth1 __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depInA __attribute__((deprecated));
- (void) depInA2 __attribute__((deprecated));
- (void) depInA1;
- (void) unavailMeth2 __attribute__((unavailable)); 
- (void) depunavailInA __attribute__((deprecated)) __attribute__((unavailable)); // expected-note {{has been explicitly marked unavailable here}}
- (void) depunavailInA1;
- (void)FuzzyMeth __attribute__((unavailable));
- (void)FuzzyMeth1 __attribute__((deprecated));
@end


@class C;	// expected-note 5 {{forward declaration of class here}}

void test(C *c) {
  [c depInA]; // expected-warning {{'depInA' maybe deprecated because receiver type is unknown}}
  [c unavailMeth]; // expected-warning {{'unavailMeth' maybe unavailable because receiver type is unknown}}
  [c depInA1]; // expected-warning {{'depInA1' maybe deprecated because receiver type is unknown}}
  [c unavailMeth1]; // expected-warning {{'unavailMeth1' maybe unavailable because receiver type is unknown}}
  [c depInA2]; // expected-warning {{'depInA2' maybe deprecated because receiver type is unknown}}
  [c unavailMeth2]; // expected-warning {{'unavailMeth2' maybe unavailable because receiver type is unknown}}
  [c depunavailInA]; // expected-warning {{'depunavailInA' maybe unavailable because receiver type is unknown}} 
  [c depunavailInA1];// expected-warning {{'depunavailInA1' maybe unavailable because receiver type is unknown}}
  [c FuzzyMeth];      // expected-warning {{'FuzzyMeth' maybe deprecated because receiver type is unknown}}
  [c FuzzyMeth1]; // expected-warning {{'FuzzyMeth1' maybe deprecated because receiver type is unknown}}

}

// rdar://10268422
__attribute ((deprecated))
@interface DEPRECATED // expected-note {{declared here}}
+(id)new;
@end

void foo() {
  [DEPRECATED new]; // expected-warning {{'DEPRECATED' is deprecated}}
}

