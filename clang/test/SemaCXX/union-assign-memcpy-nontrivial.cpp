// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fsyntax-only \
// RUN:   -Wnontrivial-memcall -verify %s

struct NonTrivialDtor {
  ~NonTrivialDtor();
};

union U {
  NonTrivialDtor n;
  int i;
};

// Odr-use both defaulted assignment operators so their bodies are synthesized.
// The synthesized memcpy must not warn.
auto get_copy = static_cast<U &(U::*)(const U &)>(&U::operator=);
auto get_move = static_cast<U &(U::*)(U &&)>(&U::operator=);

// A user-written memcpy of the same union is not suppressed and still warns.
void user_memcpy(U *d, const U *s) {
  __builtin_memcpy(d, s, sizeof(U)); // expected-warning {{first argument in call to '__builtin_memcpy' is a pointer to non-trivially copyable type 'U'}} expected-note {{explicitly cast the pointer to silence this warning}}
}
