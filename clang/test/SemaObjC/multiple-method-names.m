// RUN: %clang_cc1 -Wobjc-multiple-method-names -x objective-c %s -verify
// PR22047

@interface Face0
- (void)foo:(float)i; // expected-note {{using}}
@end

@interface Face1
- (void)foo:(int)i __attribute__((unavailable));
@end

@interface Face2
- (void)foo:(char)i; // expected-note {{also found}}
@end

void f(id i) {
  [i foo:4.0f]; // expected-warning {{multiple methods named 'foo:' found}}
}

