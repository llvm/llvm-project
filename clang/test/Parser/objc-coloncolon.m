// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify %s

@interface A
- (instancetype)init:(::A *) foo; // expected-error {{expected a type}} 
@end
