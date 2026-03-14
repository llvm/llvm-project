// RUN: %clang_cc1 -fsyntax-only -Wsigned-enum-bitfield -verify %s --std=c++11

// Enums used in bitfields with no explicitly specified underlying type.
void test0() {
  enum E { E1, E2 };
  enum F { F1, F2 };
  struct { E e1 : 1; E e2; F f1 : 1; F f2; } s;

  s.e1 = E1; // expected-warning {{enums in the Microsoft ABI are signed integers by default; consider giving the enum 'E' an unsigned underlying type to make this code portable}}
  s.f1 = F1; // expected-warning {{enums in the Microsoft ABI are signed integers by default; consider giving the enum 'F' an unsigned underlying type to make this code portable}}

  s.e2 = E2;
  s.f2 = F2;
}

// Enums used in bitfields with an explicit signed underlying type.
void test1() {
  enum E : signed { E1, E2 };
  enum F : long { F1, F2 };
  struct { E e1 : 1; E e2; F f1 : 1; F f2; } s;

  s.e1 = E1;
  s.f1 = F1;

  s.e2 = E2;
  s.f2 = F2;
}

// Enums used in bitfields with an explicitly unsigned underlying type.
void test3() {
  enum E : unsigned { E1, E2 };
  enum F : unsigned long { F1, F2 };
  struct { E e1 : 1; E e2; F f1 : 1; F f2; } s;

  s.e1 = E1;
  s.f1 = F1;

  s.e2 = E2;
  s.f2 = F2;
}
