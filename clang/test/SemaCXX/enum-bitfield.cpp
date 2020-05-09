// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify -triple x86_64-apple-darwin %s

enum E {};

struct Z {};
typedef int Integer;

struct X {
  enum E : 1; // expected-error{{anonymous bit-field}}
  enum E : Z; // expected-error{{invalid underlying type}}
  enum E2 : int;
  enum E3 : Integer;
};

struct Y {
  enum E : int(2); // expected-error{{anonymous bit-field}}
  enum E : Z(); // expected-error{{anonymous bit-field}} expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'Z'}}
};

namespace pr18587 {
struct A {
  enum class B {
    C
  };
};
const int C = 4;
struct D {
  A::B : C;
};
}

enum WithUnderlying : unsigned { wu_value };
struct WithUnderlyingBitfield {
  WithUnderlying wu : 3;
} wu = { wu_value };
int want_unsigned(unsigned);
int want_unsigned(int) = delete;
int check_enum_bitfield_promotes_correctly = want_unsigned(wu.wu);
