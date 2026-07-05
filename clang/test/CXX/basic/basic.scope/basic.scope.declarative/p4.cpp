// RUN: %clang_cc1 -std=c++2a -verify %s

namespace TagVs {
  struct Bindable { int a; };
  struct binding_a {}; // expected-note {{previous}}
  auto [binding_a] = Bindable{}; // expected-error {{redefinition}}
  auto [binding_b] = Bindable{}; // expected-note {{previous}}
  struct binding_b {}; // expected-error {{redefinition}}

  struct vartemplate_a {}; // expected-note {{previous}}
  template<typename T> int vartemplate_a; // expected-error {{redefinition}}
  template<typename T> int vartemplate_b; // expected-note {{previous}}
  struct vartemplate_b {}; // expected-error {{redefinition}}

  struct aliastemplate_a {}; // expected-note {{previous}}
  template<typename T> using aliastemplate_a = int; // expected-error {{redefinition}}
  template<typename T> using aliastemplate_b = int; // expected-note {{previous}}
  struct aliastemplate_b {}; // expected-error {{redefinition}}
}
