// RUN: %clang_cc1 -verify -Wno-objc-root-class %s

@interface XX

- (void)addObserver:(XX*)o; // expected-note 2{{passing argument to parameter 'o' here}}

@end

@interface YY

+ (void)classMethod;

@end

@implementation YY

static XX *obj;

+ (void)classMethod {
  [obj addObserver:self];     // expected-error {{cannot initialize a parameter of type 'XX *' with an lvalue of type 'Class'}}
  Class whatever;
  [obj addObserver:whatever]; // expected-error {{cannot initialize a parameter of type 'XX *' with an lvalue of type 'Class'}}
}
@end

