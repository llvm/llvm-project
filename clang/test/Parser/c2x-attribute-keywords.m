// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify %s

enum __arm_streaming E1 : int; // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

@interface Base
@end

@interface S : Base
- (void) bar;
@end

@interface T : Base
- (S *) foo;
@end


void f(T *t) {
  __arm_streaming[[t foo] bar]; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
}
