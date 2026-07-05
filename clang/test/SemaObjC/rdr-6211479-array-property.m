// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int T[2];

@interface A
@property(assign) T p2; // expected-error {{property cannot have array or function type 'T' (aka 'int[2]')}}
@end
