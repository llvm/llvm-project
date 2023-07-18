// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s
// rdar://18013929

@protocol P
- (void)meth;
@end

@interface I <P>
@end

@interface I(cat)
- (void)meth __attribute__((deprecated)); // expected-note {{'meth' has been explicitly marked deprecated here}}
@end

void foo(I *i) {
  [i meth]; // expected-warning {{'meth' is deprecated}}
}
