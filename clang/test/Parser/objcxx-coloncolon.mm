// Test to make sure the parser does not get stuck on the optional
// scope specifier on the type B.
// RUN: %clang_cc1 -fsyntax-only %s

class B;

@interface A
- (void) init:(::B *) foo;
@end
