// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -fsyntax-only -verify %s

@interface A
@end

void f0(__strong A**); // expected-note{{passing argument to parameter here}}

void test_f0(void) {
  A *a;
  static __weak A *a2;
  f0(&a);
  f0(&a2); // expected-warning{{passing 'A *__weak *' to parameter of type 'A *__strong *' discards qualifiers}}
}

void f1(__weak A**); // expected-note{{passing argument to parameter here}}

void test_f1(void) {
  A *a;
  __strong A *a2;
  f1(&a);
  f1(&a2); // expected-warning{{passing 'A *__strong *' to parameter of type 'A *__weak *' discards qualifiers}}
}

// These qualifiers should silently expand to nothing in GC mode.
void test_unsafe_unretained(__unsafe_unretained id *x) {}
void test_autoreleasing(__autoreleasing id *x) {}
