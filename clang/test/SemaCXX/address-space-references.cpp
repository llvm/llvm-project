// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int __attribute__((address_space(1))) int_1;
typedef int __attribute__((address_space(2))) int_2;

void f0(int_1 &); // expected-note{{candidate function not viable: address space mismatch in 1st argument ('int'), parameter type must be 'int_1 &' (aka '__attribute__((address_space(1))) int &')}} \
// expected-note{{candidate function not viable: address space mismatch in 1st argument ('int_2' (aka '__attribute__((address_space(2))) int')), parameter type must be 'int_1 &' (aka '__attribute__((address_space(1))) int &')}}
void f0(const int_1 &); // expected-note{{candidate function not viable: address space mismatch in 1st argument ('int'), parameter type must be 'const int_1 &' (aka 'const __attribute__((address_space(1))) int &')}} \
// expected-note{{candidate function not viable: address space mismatch in 1st argument ('int_2' (aka '__attribute__((address_space(2))) int')), parameter type must be 'const int_1 &' (aka 'const __attribute__((address_space(1))) int &')}}

void test_f0() {
  int i;
  static int_1 i1;
  static int_2 i2;

  f0(i); // expected-error{{no matching function for call to 'f0'}}
  f0(i1);
  f0(i2); // expected-error{{no matching function for call to 'f0'}}
}
