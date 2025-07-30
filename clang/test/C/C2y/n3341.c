// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=gnu -Wall -pedantic %s

/* WG14 N3341: Yes
 * Slay Some Earthly Demons III
 *
 * Empty structure and union objects are now implementation-defined.
 */

// expected-no-diagnostics

struct R {};               // gnu-warning {{empty struct is a GNU extension}}
#if __STDC_VERSION__ >= 201112L
struct S { struct { }; };  // gnu-warning {{empty struct is a GNU extension}}
#endif
struct T { int : 0; };     // gnu-warning {{struct without named members is a GNU extension}}
union U {};                // gnu-warning {{empty union is a GNU extension}}

