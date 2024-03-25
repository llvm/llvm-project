// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-unused -Wunsequenced -verify %s

struct A {
  int x, y;
};

void test() {
  int a = 0;

  A agg1( a++, a++ ); // no warning
  A agg2( a++ + a, a++ ); // expected-warning {{unsequenced modification and access to 'a'}}

  int arr1[]( a++, a++ ); // no warning
  int arr2[]( a++ + a, a++ ); // expected-warning {{unsequenced modification and access to 'a'}}
}
