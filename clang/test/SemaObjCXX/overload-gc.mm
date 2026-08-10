// RUN: %clang_cc1 -fsyntax-only -triple i386-apple-darwin9 -fobjc-gc -verify %s
// expected-no-diagnostics

void f0(__weak id *);

void test_f0(id *x) {
  f0(x);
}

@interface A
@end

void f1(__weak id*);
void test_f1(__weak A** a) {
  f1(a);
}

@interface B : A
@end

void f2(__weak A**);
void test_f2(__weak B** b) {
  f2(b);
}

