// RUN: %clang_cc1 -pedantic -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -fixit -x objective-c %t
// RUN: %clang_cc1 -pedantic -Werror -x objective-c %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

@protocol X;

void foo() {
  <X> *P;    // expected-warning{{protocol qualifiers without 'id' is archaic}}
}

@class A;
@class NSString;

@interface Test
- (void)test:(NSString *)string; // expected-note{{passing argument to parameter 'string' here}}

@property (copy) NSString *property;
@end

void g(NSString *a); // expected-note{{passing argument to parameter 'a' here}}
void h(id a); // expected-note 2{{passing argument to parameter 'a' here}}

void f(Test *t) {
  NSString *a = "Foo"; // expected-warning {{incompatible pointer types initializing 'NSString *' with an expression of type 'char [4]'}}
  id b = "Foo"; // expected-warning {{incompatible pointer types initializing 'id' with an expression of type 'char [4]'}}
  g("Foo"); // expected-warning{{incompatible pointer types passing 'char [4]' to parameter of type 'NSString *'}}
  h("Foo"); // expected-warning{{incompatible pointer types passing 'char [4]' to parameter of type 'id'}}
  h(("Foo")); // expected-warning{{incompatible pointer types passing 'char [4]' to parameter of type 'id'}}
  [t test:"Foo"]; // expected-warning{{incompatible pointer types sending 'char [4]' to parameter of type 'NSString *'}}
  t.property = "Foo"; // expected-warning{{incompatible pointer types assigning to 'NSString *' from 'char [4]'}}

  // <rdar://problem/6896493>
  [t test:@"Foo"]]; // expected-error{{extraneous ']' before ';'}}
  g(@"Foo")); // expected-error{{extraneous ')' before ';'}}
}

// rdar://7861841
@interface Radar7861841 {
@public
  int x;
}

@property (assign) int y;
@end

int f0(Radar7861841 *a) { return a.x; } // expected-error {{property 'x' not found on object of type 'Radar7861841 *'; did you mean to access instance variable 'x'}}

int f1(Radar7861841 *a) { return a->y; } // expected-error {{property 'y' found on object of type 'Radar7861841 *'; did you mean to access it with the "." operator?}}


#define nil ((void*)0)
#define NULL ((void*)0)

void sentinel(int x, ...) __attribute__((sentinel)); // expected-note{{function has been explicitly marked sentinel here}}

@interface Sentinel
- (void)sentinel:(int)x, ... __attribute__((sentinel)); // expected-note{{method has been explicitly marked sentinel here}}
@end

void sentinel_test(Sentinel *a) {
  sentinel(1, 2, 3); // expected-warning{{missing sentinel in function call}}
  [a sentinel:1, 2, 3]; // expected-warning{{missing sentinel in method dispatch}}
}
