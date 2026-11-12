// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

union u { int i; unsigned : 3; };
void f(union u);

void test(int x) {
  f((union u)x); // expected-warning {{cast to union type is a GNU extension}}
  f((union u)&x); // expected-error {{cast to union type from type 'int *' not present in union}}
  f((union u)2U); // expected-error {{cast to union type from type 'unsigned int' not present in union}}
}

union u w = (union u)2; // expected-warning {{cast to union type is a GNU extension}}
union u ww = (union u)1.0; // expected-error{{cast to union type from type 'double' not present in union}}
union u x = 7; // expected-error{{initializing 'union u' with an expression of incompatible type 'int'}}
int i;
union u zz = (union u)i; // expected-error{{initializer element is not a compile-time constant}}  expected-warning {{cast to union type is a GNU extension}}

struct s {int a, b;};
struct s y = { 1, 5 };
struct s z = (struct s){ 1, 5 };
