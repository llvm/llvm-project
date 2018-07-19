// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int __attribute__((address_space(1))) int_1;
typedef int __attribute__((address_space(2))) int_2;

void f0(int_1 &); // expected-note{{candidate function not viable: 1st argument ('int') is in address space 0, but parameter must be in address space 1}} \
// expected-note{{candidate function not viable: 1st argument ('int_2' (aka '__attribute__((address_space(2))) int')) is in address space 2, but parameter must be in address space 1}}
void f0(const int_1 &); // expected-note{{candidate function not viable: 1st argument ('int') is in address space 0, but parameter must be in address space 1}} \
// expected-note{{candidate function not viable: 1st argument ('int_2' (aka '__attribute__((address_space(2))) int')) is in address space 2, but parameter must be in address space 1}}

void test_f0() {
  int i;
  static int_1 i1;
  static int_2 i2;

  f0(i); // expected-error{{no matching function for call to 'f0'}}
  f0(i1);
  f0(i2); // expected-error{{no matching function for call to 'f0'}}
}
