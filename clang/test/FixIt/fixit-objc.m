// RUN: %clang_cc1 -pedantic -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -fixit -x objective-c %t
// RUN: %clang_cc1 -pedantic -Werror -x objective-c %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

@protocol X;

void foo(void) {
  <X> *P;    // expected-warning{{protocol has no object type specified; defaults to qualified 'id'}}
}

@class A;
@class NSString;

@interface Test
- (void)test:(NSString *)string;

@property (copy) NSString *property;
@end

void g(NSString *a);
void h(id a);

void f(Test *t) {
  NSString *a = "Foo"; // expected-error {{string literal must be prefixed by '@'}}
  id b = "Foo"; // expected-error {{string literal must be prefixed by '@'}}
  g("Foo"); // expected-error {{string literal must be prefixed by '@'}}
  h("Foo"); // expected-error {{string literal must be prefixed by '@'}}
  h(("Foo")); // expected-error {{string literal must be prefixed by '@'}}
  [t test:"Foo"]; // expected-error {{string literal must be prefixed by '@'}}
  t.property = "Foo"; // expected-error {{string literal must be prefixed by '@'}}

  [t test:@"Foo"]]; // expected-error{{extraneous ']' before ';'}}
  g(@"Foo")); // expected-error{{extraneous ')' before ';'}}
}

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

@interface A
@property (class) int c;
@end

int test(A *a) {
  return a.c; // expected-error {{property 'c' is a class property; did you mean to access it with class 'A'}}
}
