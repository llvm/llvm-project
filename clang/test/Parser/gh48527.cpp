// RUN: %clang_cc1 -fsyntax-only -verify %s

int main() { // expected-note {{to match this '{'}}
    auto a = [](void)__attribute__((b(({ // expected-note {{to match this '('}}
    return 0;
} // expected-error 3 {{expected ')'}} \
  // expected-error {{expected ';' at end of declaration}}
// expected-error@+2 {{expected ')'}}
// expected-error@+1 {{expected body of lambda expression}}
// expected-error {{expected '}'}}
