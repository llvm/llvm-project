// RUN: %clang_cc1 -fsyntax-only -verify %s
struct foo {
  int a;
};

int main(void) {
  struct foo xxx;
  int i;

  xxx = (struct foo)1;  // expected-error {{used type 'struct foo' where arithmetic or pointer type is required}}
  i = (int)xxx; // expected-error {{operand of type 'struct foo' where arithmetic or pointer type is required}}
}
