// RUN: %clang_cc1 %s -verify -fsyntax-only

extern const int W1 __attribute__((weak)) = 10; // expected-note {{declared here}}

static_assert(W1 == 10, ""); // expected-error   {{static assertion expression is not an integral constant expression}}
                             // expected-note@-1 {{initializer of weak variable 'W1' is not considered constant because it may be different at runtime}}

extern const int W2 __attribute__((weak)) = 20;

int S2[W2]; // expected-error {{variable length array declaration not allowed at file scope}}

extern const int W3 __attribute__((weak)) = 30; // expected-note {{declared here}}

constexpr int S3 = W3; // expected-error   {{constexpr variable 'S3' must be initialized by a constant expression}}
                       // expected-note@-1 {{initializer of weak variable 'W3' is not considered constant because it may be different at runtime}}
