// RUN: %clang_cc1 -std=c++26 -verify %s

template <typename>
consteval {} // expected-error {{a consteval block declaration cannot be a template}}

struct X0 {
  template <typename>
  consteval {} // expected-error {{a consteval block declaration cannot be a template}}
};

void f1() {
  template <typename>
  consteval {} // expected-error {{a consteval block declaration cannot be a template}}
}

inline consteval {} // expected-error {{expected unqualified-id}}
static consteval {} // expected-error {{expected unqualified-id}}
consteval inline {} // expected-error {{expected unqualified-id}}
consteval static {} // expected-error {{expected unqualified-id}}

struct X1 { inline consteval {} }; // expected-error {{expected member name or ';' after declaration specifiers}}
struct X2 { static consteval {} }; // expected-error {{expected member name or ';' after declaration specifiers}}
struct X3 { consteval inline {} }; // expected-error {{expected member name or ';' after declaration specifiers}}
struct X4 { consteval static {} }; // expected-error {{expected member name or ';' after declaration specifiers}}

void f2() {
  inline consteval {} // expected-error {{expected unqualified-id}}
  static consteval {} // expected-error {{expected unqualified-id}}
  consteval inline {} // expected-error {{expected unqualified-id}}
  consteval static {} // expected-error {{expected unqualified-id}}
}
