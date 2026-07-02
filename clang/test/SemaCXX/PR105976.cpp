// RUN: %clang_cc1 -fsyntax-only -verify -verify-ignore-unexpected=note %s

struct A1 {
  void operator==(A1);
};

struct A2 {
  void operator==(A2);
};

struct A3 {
  void operator==(A3);
};

struct A4 : A1, A2 {
  using A1::operator==;
  using A2::operator==;
};

struct A5 : A3, A4 {};

void A6() {
  A5{} == A5{};
  // expected-error@-1 {{member 'operator==' found in multiple base classes of different types}}
  // expected-error@-2 {{use of overloaded operator '==' is ambiguous}}
}


struct B1 {
  template <typename T> void operator==(T);
};

struct B2 {
  template <typename T> void operator==(T);
};

struct B3 {
  template <typename T> void operator==(T);
};

struct B4 : B1, B2 {
  using B1::operator==;
  using B2::operator==;
};

struct B5 : B3, B4 {};

void B6() {
  B5{} == B5{};
  // expected-error@-1 {{member 'operator==' found in multiple base classes of different types}}
  // expected-error@-2 {{use of overloaded operator '==' is ambiguous}}
}
