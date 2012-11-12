// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -fsyntax-only -verify %s

@interface A
@end

void f0(__strong A**); // expected-note{{candidate function not viable: 1st argument ('A *__weak *') has __weak ownership, but parameter has __strong ownership}}

void test_f0() {
  A *a;
  static __weak A *a2;
  f0(&a);
  f0(&a2); // expected-error{{no matching function}}
}

void f1(__weak A**); // expected-note{{candidate function not viable: 1st argument ('A *__strong *') has __strong ownership, but parameter has __weak ownership}}

void test_f1() {
  A *a;
  __strong A *a2;
  f1(&a);
  f1(&a2); // expected-error{{no matching function}}
}
