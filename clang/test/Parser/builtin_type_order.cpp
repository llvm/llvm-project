// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename... T>
void syntax_errors() {
  (void)__builtin_type_order(int); // expected-error {{expected ','}}
  (void)__builtin_type_order(T...); // expected-error {{expected ','}}

  (void)__builtin_type_order(int, long, char); 
  // expected-error@-1 {{expected ')'}} \
  //   expected-note@-1 {{to match this '('}}
  (void)__builtin_type_order(int, T...);
  // expected-error@-1 {{expected ')'}} \
  //   expected-note@-1 {{to match this '('}}
}
