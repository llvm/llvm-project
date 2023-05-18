// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Foo {
  id x;
}
@property (nonatomic, retain) id x; // expected-note{{property declared here}}
@property (nonatomic, retain) id x; // expected-error{{property has a previous declaration}}
@end
