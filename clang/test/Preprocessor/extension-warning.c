// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -Wc++-keyword %s
// RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify -Wc++11-compat %s

#define ID(x) x
#define DISCARD(x)
#define EXPAND_AND_DISCARD(x) DISCARD(x)

#ifndef __cplusplus

// The preprocessor shouldn't warn about extensions within macro bodies that
// aren't expanded.
#define TY typeof
#define TY1 typeof(1)

// But we should warn here
TY1 x; // expected-warning {{extension}}
TY(1) x; // expected-warning {{extension}}

#define TY_FN() typeof
TY_FN()(1) y; // expected-warning {{extension}}

ID(TY)(1) z; // expected-warning {{extension}}
ID(TY1) w; // expected-warning {{extension}}

EXPAND_AND_DISCARD(TY)

// Note: this warning intentionally doesn't trigger on keywords like
// __attribute; the standard allows implementation-defined extensions
// prefixed with "__".
// Current list of keywords this can trigger on:
// inline, restrict, asm, typeof, _asm

void whatever(void) {}

#define CLASS class
void macro_keyword(void) {
  int CLASS; // expected-warning {{identifier 'class' conflicts with a C++ keyword}}
}

void macro_argument_keyword(void) {
  int ID(CLASS); // expected-warning {{identifier 'class' conflicts with a C++ keyword}}
}

#else

#define constexpr const
constexpr int x = 0;
#undef constexpr

#define constexpr constexpr
int constexpr; // expected-warning {{'constexpr' is a keyword in C++11}}
namespace repeated {
int constexpr; // already diagnosed in this TU
}
#undef constexpr

#define NOEXCEPT noexcept
EXPAND_AND_DISCARD(NOEXCEPT)
int ID(NOEXCEPT); // expected-warning {{'noexcept' is a keyword in C++11}}

#endif
