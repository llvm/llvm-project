// RUN: %clang_cc1 -verify=expected,pre2a %s -std=c++11
// RUN: %clang_cc1 -verify=expected,pre2a %s -std=c++17
// RUN: %clang_cc1 -verify=expected %s -std=c++2a

// A function that is explicitly defaulted shall
struct A {
  // -- be a special member function [C++2a: or a comparison operator function],
  A(int) = default;
#if __cplusplus <= 201703L
  // expected-error@-2 {{only special member functions may be defaulted}}
#else
  // expected-error@-4 {{only special member functions and comparison operators may be defaulted}}
#endif
  A(A) = default; // expected-error {{must pass its first argument by reference}}
  void f(A) = default; // expected-error-re {{only special member functions{{( and comparison operators)?}} may be defaulted}}

  bool operator==(const A&) const = default; // pre2a-warning {{defaulted comparison operators are a C++20 extension}}
  bool operator!=(const A&) const = default; // pre2a-warning {{defaulted comparison operators are a C++20 extension}}
  bool operator<(const A&) const = default; // pre2a-error {{only special member functions may be defaulted}}
  bool operator>(const A&) const = default; // pre2a-error {{only special member functions may be defaulted}}
  bool operator<=(const A&) const = default; // pre2a-error {{only special member functions may be defaulted}}
  bool operator>=(const A&) const = default; // pre2a-error {{only special member functions may be defaulted}}
  bool operator<=>(const A&) const = default; // pre2a-error 1+{{}} pre2a-warning {{'<=>' is a single token in C++2a}}

  A operator+(const A&) const = default; // expected-error-re {{only special member functions{{( and comparison operators)?}} may be defaulted}}

  // -- have the same declared function type as if it had been implicitly
  //    declared
  void operator=(const A &) = default; // expected-error {{must return 'A &'}}
  A(...) = default;
  A(const A &, ...) = default;
  A &operator=(const A&) const = default;
  A &operator=(A) const = default; // expected-error {{must be an lvalue refe}}
#if __cplusplus <= 201703L
  // expected-error@-5 {{cannot be variadic}}
  // expected-error@-5 {{cannot be variadic}}
  // expected-error@-5 {{may not have 'const'}}
  // expected-error@-5 {{may not have 'const'}}
#else
  // expected-warning@-10 {{implicitly deleted}} expected-note@-10 {{declared type does not match the type of an implicit default constructor}}
  // expected-warning@-10 {{implicitly deleted}} expected-note@-10 {{declared type does not match the type of an implicit copy constructor}}
  // expected-warning@-10 {{implicitly deleted}} expected-note@-10 {{declared type does not match the type of an implicit copy assignment}}
#endif

  //    (except for possibly differing ref-qualifiers
  A &operator=(A &&) & = default;

  //    and except that in the case of a copy constructor or copy assignment
  //    operator, the parameter type may be "reference to non-const T")
  A(A &) = default;
  A &operator=(A &) = default;

  // -- not have default arguments
  A(double = 0.0) = default; // expected-error {{cannot have default arguments}}
  A(const A & = 0) = default; // expected-error {{cannot have default arguments}}
};

struct A2 {
  A2(...);
  A2(const A2 &, ...);
  A2 &operator=(const A2&) const;
};
A2::A2(...) = default; // expected-error {{cannot be variadic}}
A2::A2(const A2&, ...) = default; // expected-error {{cannot be variadic}}
A2 &A2::operator=(const A2&) const = default; // expected-error {{may not have 'const'}}

struct B {
  B(B&);
  B &operator=(B&);
};
struct C : B {
  C(const C&) = default;
  C &operator=(const C&) = default;
#if __cplusplus <= 201703L
  // expected-error@-3 {{is const, but a member or base requires it to be non-const}}
  // expected-error@-3 {{is const, but a member or base requires it to be non-const}}
#else
  // expected-warning@-6 {{implicitly deleted}} expected-note@-6 {{type does not match}}
  // expected-warning@-6 {{implicitly deleted}} expected-note@-6 {{type does not match}}
#endif
};

struct D : B { // expected-note 2{{base class}}
  D(const D&);
  D &operator=(const D&);
};
D::D(const D&) = default; // expected-error {{would delete}} expected-error {{is const, but}}
D &D::operator=(const D&) = default; // expected-error {{would delete}} expected-error {{is const, but}}
