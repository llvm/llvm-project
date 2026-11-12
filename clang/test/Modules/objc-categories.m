// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=category_top -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=category_left -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=category_right -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=category_bottom -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=category_other -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import category_bottom;

// expected-note@Inputs/category_top.h:1 {{receiver is instance of class declared here}}

@interface Foo(Source)
-(void)source; 
@end

void test(Foo *foo, LeftFoo *leftFoo) {
  [foo source];
  [foo bottom];
  [foo left];
  [foo right1];
  [foo right2];
  [foo top];
  [foo top2];
  [foo top3];

  [leftFoo left];
  [leftFoo bottom];
}

// Load another module that also adds categories to Foo, verify that
// we see those categories.
@import category_other;

void test_other(Foo *foo) {
  [foo other];
}

// Make sure we don't see categories that should be hidden
void test_hidden_all_errors(Foo *foo) {
  [foo left_sub]; // expected-warning{{instance method '-left_sub' not found (return type defaults to 'id')}}
  foo.right_sub_prop = foo; // expected-error{{property 'right_sub_prop' not found on object of type 'Foo *'}}
  int i = foo->right_sub_ivar; // expected-error{{'Foo' does not have a member named 'right_sub_ivar'}}
  id<P1> p1 = foo; // expected-warning{{initializing 'id<P1>' with an expression of incompatible type 'Foo *'}}
  id<P2> p2 = foo; // expected-warning{{initializing 'id<P2>' with an expression of incompatible type 'Foo *'}}
  id<P3> p3;
  [p3 p3_method]; // expected-warning{{instance method '-p3_method' not found (return type defaults to 'id')}}
  id<P4> p4;
  [p4 p4_method]; // expected-warning{{instance method '-p4_method' not found (return type defaults to 'id')}}
  id p3p = p3.p3_prop; // expected-error{{property 'p3_prop' not found on object of type 'id<P3>'}}
  p3p = foo.p3_prop; // expected-error{{property 'p3_prop' not found on object of type 'Foo *'}}
  id p4p = p4.p4_prop; // expected-error{{property 'p4_prop' not found on object of type 'id<P4>'}}
  p4p = foo.p4_prop; // expected-error{{property 'p4_prop' not found on object of type 'Foo *'}}

  if (foo.hiddenPropertyFromExtension) { // expected-error {{property 'hiddenPropertyFromExtension' not found on object of type 'Foo *'}}
  }
}

@import category_left.sub;

void test_hidden_right_errors(Foo *foo) {
  // These are okay
  [foo left_sub]; // okay
  id<P1> p1 = foo;
  id<P3> p3;
  [p3 p3_method];
  id p3p = p3.p3_prop;
  p3p = foo.p3_prop;
  // These should fail
  foo.right_sub_prop = foo; // expected-error{{property 'right_sub_prop' not found on object of type 'Foo *'}}
  int i = foo->right_sub_ivar; // expected-error{{'Foo' does not have a member named 'right_sub_ivar'}}
  id<P2> p2 = foo; // expected-warning{{initializing 'id<P2>' with an expression of incompatible type 'Foo *'}}
  id<P4> p4;
  [p4 p4_method]; // expected-warning{{instance method '-p4_method' not found (return type defaults to 'id')}}
  id p4p = p4.p4_prop; // expected-error{{property 'p4_prop' not found on object of type 'id<P4>'}}
  p4p = foo.p4_prop; // expected-error{{property 'p4_prop' not found on object of type 'Foo *'; did you mean 'p3_prop'?}}
  // expected-note@Inputs/category_left_sub.h:7{{'p3_prop' declared here}}
  int hiddenFromExtension = foo.hiddenPropertyFromExtension; // expected-error {{property 'hiddenPropertyFromExtension' not found on object of type 'Foo *'}}
}

@import category_right.sub;

void test_hidden_okay(Foo *foo) {
  [foo left_sub];
  foo.right_sub_prop = foo;
  int i = foo->right_sub_ivar;
  id<P1> p1 = foo;
  id<P2> p2 = foo;
  id<P3> p3;
  [p3 p3_method];
  id<P4> p4;
  [p4 p4_method];
  id p3p = p3.p3_prop;
  p3p = foo.p3_prop;
  id p4p = p4.p4_prop;
  p4p = foo.p4_prop;
  if (foo.hiddenPropertyFromExtension) {
  }
}
