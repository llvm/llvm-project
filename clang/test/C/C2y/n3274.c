// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s

/* WG14 N3274: Yes
 * Remove imaginary types
 */

// Clang has never supported _Imaginary.
#ifdef __STDC_IEC_559_COMPLEX__
#error "When did this happen?"
#endif

_Imaginary float i; // expected-error {{imaginary types are not supported}}

// _Imaginary is a keyword in older language modes, but doesn't need to be one
// in C2y or later. However, to improve diagnostic behavior, we retain it as a
// keyword in all language modes -- it is not available as an identifier.
static_assert(!__is_identifier(_Imaginary));
