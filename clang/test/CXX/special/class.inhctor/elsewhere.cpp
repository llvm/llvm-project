// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Tests related to constructor inheriting, but not specified in [class.inhctor]

// [namespace.udecl]p8:
//   A using-declaration for a class member shall be a member-declaration.

struct B1 {
  B1(int);
};

using B1::B1; // expected-error {{using declaration can not refer to class member}} expected-error {{not supported}}

// C++0x [namespace.udecl]p10:
//   A using-declaration is a declaration and can therefore be used repeatedly
//   where (and only where) multiple declarations are allowed.

struct I1 : B1 {
  using B1::B1; // expected-note {{previous using declaration}} expected-error {{not supported}}
  using B1::B1; // expected-error {{redeclaration of using decl}} expected-error {{not supported}}
};

// C++0x [namespace.udecl]p3:
//   In a using declaration used as a member-declaration, the nested-name-
//   specifier shall name a base class of the class being defined.
//   If such a using-declaration names a constructor, the nested-name-specifier
//   shall name a direct base class of the class being defined.

struct D1 : I1 {
  using B1::B1; // expected-error {{'B1' is not a direct base of 'D1', can not inherit constructors}} expected-error {{not supported}}
};

template<typename T> struct A {};

template<typename T> struct B : A<bool>, A<char> {
  using A<T>::A; // expected-error {{'A<double>::', which is not a base class of 'B<double>'}} expected-error {{not supported}}
};
B<bool> bb;
B<char> bc;
B<double> bd; // expected-note {{here}}

template<typename T> struct C : A<T> {
  using A<bool>::A; // expected-error {{'A<bool>::', which is not a base class of 'C<char>'}} expected-error {{not supported}}
};
C<bool> cb;
C<char> cc; // expected-note {{here}}

template<typename T> struct D : A<T> {};
template<typename T> struct E : D<T> {
  using A<bool>::A; // expected-error {{'A<bool>' is not a direct base of 'E<bool>', can not inherit}} expected-error {{not supported}}
};
E<bool> eb; // expected-note {{here}}

template<typename T> struct F : D<bool> {
  using A<T>::A; // expected-error {{'A<bool>' is not a direct base of 'F<bool>'}} expected-error {{not supported}}
};
F<bool> fb; // expected-note {{here}}
