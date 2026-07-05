// RUN: %clang_cc1 -fsyntax-only -verify %s

@implementation I (C) // expected-error {{cannot find interface declaration for 'I'}}

+ (void)f {
  self.m; // expected-error {{member reference base type 'Class' is not a structure or union}}
}

@end
