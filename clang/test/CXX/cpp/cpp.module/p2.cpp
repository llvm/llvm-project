// RUN: %clang_cc1 -std=c++2a -emit-header-module -fmodule-name=attrs -x c++-header %S/Inputs/empty.h %S/Inputs/attrs.h -o %t.pcm
// RUN: %clang_cc1 -std=c++2a %s -fmodule-file=%t.pcm -fsyntax-only -verify -I%S/Inputs

template<int> struct import; // expected-note 2{{previous}}
constexpr struct { int h; } empty = {0};
struct A;
struct B;
struct C;
template<> struct import<0> {
  static A a;
  static B b;
  static C c;
};

// OK, not an import-declaration.
// FIXME: This is valid, see PR41192
struct A {} // FIXME expected-error {{expected ';'}}
::import
<empty.h>::a; // FIXME expected-error {{requires a type specifier}}

// This is invalid: the tokens after 'import' are a header-name, so cannot be
// parsed as a template-argument-list.
struct B {}
import // expected-error {{redefinition of 'import'}} expected-error {{expected ';'}}
<empty.h>::b; // (error recovery skips these tokens)

// Likewise, this is ill-formed after the tokens are reconstituted into a
// header-name token.
struct C {}
import // expected-error {{redefinition of 'import'}} expected-error {{expected ';'}}
<
empty.h // (error recovery skips these tokens)
>::c;
