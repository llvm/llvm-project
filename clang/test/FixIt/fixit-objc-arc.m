// RUN: %clang_cc1 -pedantic -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -fobjc-arc -fixit -x objective-c %t
// RUN: %clang_cc1 -pedantic -fobjc-arc -Werror -x objective-c %t
// rdar://14106083

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
  g("Foo"); // expected-error {{string literal must be prefixed by '@'}}
  [t test:"Foo"]; // expected-error {{string literal must be prefixed by '@'}}
  t.property = "Foo"; // expected-error {{string literal must be prefixed by '@'}}
}
