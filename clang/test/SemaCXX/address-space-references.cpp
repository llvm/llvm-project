// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int __attribute__((address_space(1))) int_1;
typedef int __attribute__((address_space(2))) int_2;

void f0(int_1 &);       // expected-note{{candidate function not viable: cannot bind reference in generic address space to object in address space '1' in 1st argument}} \
// expected-note{{candidate function not viable: cannot bind reference in address space '2' to object in address space '1' in 1st argument}}
void f0(const int_1 &); // expected-note{{candidate function not viable: cannot bind reference in generic address space to object in address space '1' in 1st argument}} \
// expected-note{{candidate function not viable: cannot bind reference in address space '2' to object in address space '1' in 1st argument}}

void test_f0() {
  int i;
  static int_1 i1;
  static int_2 i2;

  f0(i); // expected-error{{no matching function for call to 'f0'}}
  f0(i1);
  f0(i2); // expected-error{{no matching function for call to 'f0'}}
}
