// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s

constexpr const int *foo[][2] = { {nullptr, int}, }; // expected-error {{expected '(' for function-style cast or type construction}} \
                                                     // expected-note {{declared here}}
static_assert(foo[0][0] == nullptr, ""); // expected-error {{constant expression}} \
                                         // expected-note {{initializer of 'foo' is unknown}}
