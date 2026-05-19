// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify

// [cpp.cond]/p9: If the preprocessing token defined is generated as a result
// of this replacement process, the program is ill-formed, no diagnostic
// required.
#define DEFINED defined
#if DEFINED(bar) // expected-warning {{macro expansion producing 'defined' has undefined behavior}}
#endif

// Check that a real-world system header pattern remains accepted.
#include "Inputs/WinBase.h"

// [cpp.cond]/p9: If use of the defined unary operator does not match one of
// the two specified forms prior to macro replacement, the program is
// ill-formed, no diagnostic required.
#if defined() // expected-error {{macro name must be an identifier}}
#endif
#if defined(a b) // expected-error {{missing ')' after 'defined'}} expected-note {{to match this '('}}
#endif
#if defined(a, b) // expected-error {{missing ')' after 'defined'}} expected-note {{to match this '('}}
#endif
