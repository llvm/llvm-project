// RUN: %clang_cc1 -DFIRST -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSECOND -fsyntax-only -verify %s
// RUN: %clang_cc1 -DTHIRD -fsyntax-only -verify %s
// RUN: %clang_cc1 -DFOURTH -fsyntax-only -verify %s

@protocol P;
@interface NSObject
@end
@protocol X
@end
@interface X : NSObject <X>
@end

@class A;

#ifdef FIRST
id<X> F1(id<[P> v) { // expected-error {{expected a type}} // expected-error {{use of undeclared identifier 'P'}} // expected-error {{use of undeclared identifier 'v'}} // expected-note {{to match this '('}}
  return 0;
}
#endif

#ifdef SECOND
id<X> F2(id<P[X> v) { // expected-error {{unknown type name 'P'}} // expected-error {{unexpected interface name 'X': expected expression}} // expected-error {{use of undeclared identifier 'v'}} // expected-note {{to match this '('}}
  return 0;
}
#endif

#ifdef THIRD
id<X> F3(id<P, P *[> v) { // expected-error {{unknown type name 'P'}} // expected-error {{expected expression}} // expected-error {{use of undeclared identifier 'v'}} // expected-note {{to match this '('}}
  return 0;
}
#endif

#ifdef FOURTH
id<X> F4(id<P, P *(> v { // expected-error {{unknown type name 'P'}} // expected-error {{expected ')'}} // expected-note {{to match this '('}} // expected-note {{to match this '('}}
  return 0;
}
#endif

// expected-error {{expected '>'}} // expected-error {{expected parameter declarator}} // expected-error {{expected ')'}} // expected-error {{expected function body after function declarator}}
