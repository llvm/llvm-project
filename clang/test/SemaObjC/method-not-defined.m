// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Foo
@end

void test() {
  Foo *fooObj;
  id obj;

  [[Foo alloc] init]; // expected-warning {{class method '+alloc' not found (return type defaults to 'id')}} expected-warning {{instance method '-init' not found (return type defaults to 'id')}}
  [fooObj notdefined]; // expected-warning {{instance method '-notdefined' not found (return type defaults to 'id')}}
  [obj whatever:1 :2 :3]; // expected-warning {{instance method '-whatever:::' not found (return type defaults to 'id')}}
}
