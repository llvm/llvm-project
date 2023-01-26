// RUN: %clang_cc1 -verify %s

/* WG14 N1356: Yes
 * _Bool bit-fields
 */

// C does not allow the bit-width of a bit-field to exceed the width of the
// bit-field type, and _Bool is only required to be able to store 0 and 1
// (and thus is implicitly unsigned), which only requires a single bit.
#if __BOOL_WIDTH__ < __CHAR_BIT__
struct S {
  _Bool b : __CHAR_BIT__; // expected-error {{width of bit-field 'b' (8 bits) exceeds the width of its type (1 bit)}}
};
#else
// expected-no-diagnostics
#endif

