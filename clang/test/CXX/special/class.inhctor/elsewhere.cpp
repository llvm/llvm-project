// RUN: true
// Disabled for 2.9
//%clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Tests related to constructor inheriting, but not specified in [class.inhctor]

// [namespace.udecl]p8:
//   A using-declaration for a class member shall be a member-declaration.

struct B1 {
  B1(int);
};

using B1::B1; // expected-error {{using declaration can not refer to class member}}

// C++0x [namespace.udecl]p10:
//   A using-declaration is a declaration and can therefore be used repeatedly
//   where (and only where) multiple declarations are allowed.

struct I1 : B1 {
  using B1::B1; // expected-note {{previous using declaration}}
  using B1::B1; // expected-error {{redeclaration of using decl}}
};

// C++0x [namespace.udecl]p3:
//   In a using declaration used as a member-declaration, the nested-name-
//   specifier shall name a base class of the class being defined.
//   If such a using-declaration names a constructor, the nested-name-specifier
//   shall name a direct base class of the class being defined.

struct D1 : I1 {
  using B1::B1; // expected-error {{'B1' is not a direct base of 'D1', can not inherit constructors}}
};
