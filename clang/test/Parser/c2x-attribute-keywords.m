// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify %s

enum __arm_inout("za") E1 : int; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

@interface Base
@end

@interface S : Base
- (void) bar;
@end

@interface T : Base
- (S *) foo;
@end


void f(T *t) {
  __arm_inout("za")[[t foo] bar]; // expected-error {{'__arm_inout' cannot be applied to a statement}}
}
