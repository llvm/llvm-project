// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

_Alignas(int) struct c1; // expected-warning {{'_Alignas' attribute ignored}}

// FIXME: `alignas` enters into C++ parsing code and never reaches the
// declaration specifier attribute diagnostic infrastructure.
// 
// Fixing this will require the C23 notions of `alignas` being a keyword and
// `_Alignas` being an alternate spelling integrated into the parsing
// infrastructure.
alignas(int) struct c1; // expected-error {{misplaced attributes; expected attributes here}}
