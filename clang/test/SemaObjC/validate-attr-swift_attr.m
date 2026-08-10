// RUN: %clang_cc1 -verify -fsyntax-only %s

// expected-error@+1 {{'swift_attr' attribute takes one argument}}
__attribute__((swift_attr))
@interface I
@end

// expected-error@+1 {{expected string literal as argument of 'swift_attr' attribute}}
__attribute__((swift_attr(1)))
@interface J
@end

@interface Error<T: __attribute__((swift_attr(1))) id>
// expected-error@-1 {{expected string literal as argument of 'swift_attr' attribute}}
@end
