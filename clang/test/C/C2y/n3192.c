// RUN: %clang_cc1 -verify -Wno-c23-extensions %s

/* WG14 N3192: Yes
 * Sequential hexdigits
 */

// expected-no-diagnostics

// Demonstrate that hex digits are already sequential in all targets Clang
// supports.

#define TEST_VAL(ch) ((ch >= 'A' && ch <= 'F') || (ch >= 'a' && ch <= 'f'))
#define GET_VAL(ch)  (((ch >= 'A' && ch <= 'F') ? (ch - 'A') : (ch - 'a')) + 10)

_Static_assert(TEST_VAL('A'));
_Static_assert(TEST_VAL('B'));
_Static_assert(TEST_VAL('C'));
_Static_assert(TEST_VAL('D'));
_Static_assert(TEST_VAL('E'));
_Static_assert(TEST_VAL('F'));
_Static_assert(TEST_VAL('a'));
_Static_assert(TEST_VAL('b'));
_Static_assert(TEST_VAL('c'));
_Static_assert(TEST_VAL('d'));
_Static_assert(TEST_VAL('e'));
_Static_assert(TEST_VAL('f'));

_Static_assert(!TEST_VAL('G'));
_Static_assert(!TEST_VAL('h'));

_Static_assert(GET_VAL('A') == 0xA);
_Static_assert(GET_VAL('B') == 0xB);
_Static_assert(GET_VAL('C') == 0xC);
_Static_assert(GET_VAL('D') == 0xD);
_Static_assert(GET_VAL('E') == 0xE);
_Static_assert(GET_VAL('F') == 0xF);
_Static_assert(GET_VAL('a') == 0xA);
_Static_assert(GET_VAL('b') == 0xB);
_Static_assert(GET_VAL('c') == 0xC);
_Static_assert(GET_VAL('d') == 0xD);
_Static_assert(GET_VAL('e') == 0xE);
_Static_assert(GET_VAL('f') == 0xF);

