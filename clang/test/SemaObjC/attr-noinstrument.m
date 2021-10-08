// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10.4 -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple x86_64-apple-darwin10.4 -verify -Wno-objc-root-class %s

// expected-no-diagnostics
@interface A
+ (void)F __attribute__((no_instrument_function)); // no warning
- (void)f __attribute__((objc_direct, no_instrument_function));
- (void)g;
@end

@implementation A
+ (void)F __attribute__((no_instrument_function)) {
  [self F];
}

- (void)f {
  [self g];
}

- (void)g {
}
@end
