// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P // expected-note {{previous}}
-(void)meth1;
@end

@interface I // expected-note {{previous}}
@end

@interface I2
@end
@interface I2(C) // expected-note {{previous}}
@end

extern "C" {
@protocol P // expected-warning {{duplicate protocol definition of 'P' is ignored}}
-(void)meth2;
@end

@interface I // expected-error {{duplicate}}
@end

@interface I2(C) // expected-warning {{duplicate}}
@end
}

void test(id<P> p) {
  [p meth1];
  [p meth2]; // expected-warning {{not found}}
}
