// RUN: %clang_cc1 -verify -Wno-objc-root-class %s

@interface ClassA

- (void)fileExistsAtPath:(int)x;

@end

@interface ClassB

@end

@implementation ClassB // expected-note {{implementation started here}}

- (void) method:(ClassA *)mgr { // expected-note {{to match this '{'}}
  [mgr fileExistsAtPath:0
} // expected-error {{expected ']'}}

@interface ClassC //                   \
  // expected-error {{missing '@end'}} \
  // expected-error {{expected '}'}}

@end
