// RUN: %clang_analyze_cc1 -w -Wno-int-conversion %s -verify \
// RUN:     -analyzer-checker=core,alpha.core,debug.ExprInspection

#ifdef HEADER // A clever trick to avoid splitting up the test.
extern void clang_analyzer_eval(int);

@interface NSObject
@end

@interface HeaderClass : NSObject
@property NSObject *prop;
@end

#else
#define HEADER
#include "ObjCProperties.m"

@implementation HeaderClass
- (void)foo {
  if ((self.prop)) {
  }

  // This test tests that no dynamic bifurcation is performed on the property.
  // The TRUE/FALSE dilemma correctly arises from eagerly-assume behavior
  // inside the if-statement. The dynamic bifurcation at (self.prop) inside
  // the if-statement was causing an UNKNOWN to show up as well due to
  // extra parentheses being caught inside PseudoObjectExpr.
  // This should not be UNKNOWN.
  clang_analyzer_eval(self.prop); // expected-warning{{TRUE}}
                                  // expected-warning@-1{{FALSE}}
}
@end


// The point of this test cases is to exercise properties in the static
// analyzer

@interface MyClass {
@private
    id _X;
}
- (id)initWithY:(id)Y;
@property(copy, readwrite) id X;
@end

@implementation MyClass
@synthesize X = _X;
- (id)initWithY:(id)Y {
  self.X = Y;
  return self;
}
@end
#endif
