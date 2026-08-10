// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -fblocks %s

@interface Test0
@property (assign) id x  // expected-error {{expected ';' at end of declaration list}}
@end
