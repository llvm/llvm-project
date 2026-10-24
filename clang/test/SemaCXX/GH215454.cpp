// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename T> void gh215454() {
  static_assert(__is_same(decltype(({ 1; int; })), void)); // expected-warning {{declaration does not declare anything}}
  static_assert(__is_same(decltype(({ 1; int; })), int));  // expected-warning {{declaration does not declare anything}} \
                                                           // expected-error {{static assertion failed}}
}
template void gh215454<int>(); // expected-note {{in instantiation of function template specialization 'gh215454<int>' requested here}}
