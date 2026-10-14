// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fsyntax-only \
// RUN:   -Wnontrivial-memcall -Wdeprecated-copy-with-user-provided-dtor -verify %s

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

// The memcpy suppression is scoped to the synthesized call, so an unrelated
// warning for the same union still fires: a user-provided destructor deprecates
// the implicit copy assignment.
union V {
  int i;
  ~V() {} // expected-warning {{definition of implicit copy assignment operator for 'V' is deprecated because it has a user-provided destructor}}
};

void use_deprecated_copy(V &a, const V &b) {
  a = b; // expected-note {{in implicit copy assignment operator for 'V' first required here}}
}
