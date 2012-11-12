// RUN: %clang_cc1 -verify -fsyntax-only %s

__attribute__((deprecated)) @class B; // expected-error {{prefix attribute must be followed by an interface or protocol}}

__attribute__((deprecated)) @interface A @end
__attribute__((deprecated)) @protocol P0;
__attribute__((deprecated)) @protocol P1
@end
