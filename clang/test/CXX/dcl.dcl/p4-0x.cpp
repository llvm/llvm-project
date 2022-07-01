// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only %s

struct S {
  constexpr S(bool b) : b(b) {}
  constexpr explicit operator bool() const { return b; }
  bool b;
};
struct T {
  constexpr operator int() const { return 1; }
};
struct U {
  constexpr operator int() const { return 1; } // expected-note {{candidate}}
  constexpr operator long() const { return 0; } // expected-note {{candidate}}
};

static_assert(S(true), "");
static_assert(S(false), "not so fast"); // expected-error {{not so fast}}
static_assert(T(), "");
static_assert(U(), ""); // expected-error {{ambiguous}}

static_assert(false, L"\x14hi" "!" R"x(")x"); // expected-error {{static_assert failed: L"\024hi!\""}}
