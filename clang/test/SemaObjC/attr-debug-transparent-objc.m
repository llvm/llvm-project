// RUN: %clang_cc1 %s -verify -fsyntax-only



@interface ObjCClass
- (void)correct __attribute__((debug_transparent));
- (void)one_arg __attribute__((debug_transparent(1))); // expected-error {{'debug_transparent' attribute takes no arguments}}
@end

