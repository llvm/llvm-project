// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=gnu -Wall -pedantic %s

/* WG14 N3341: Yes
 * Slay Some Earthly Demons III
 *
 * Structure and union objects with a member declaration list but no named
 * members are now implementation-defined.
 */

struct R {};               // expected-warning {{empty struct is a GNU extension}} \
                           // gnu-warning {{empty struct is a GNU extension}}
#if __STDC_VERSION__ >= 201112L
struct S { struct { }; };  // expected-warning {{empty struct is a GNU extension}} \
                           // gnu-warning {{empty struct is a GNU extension}}
#endif
struct T { int : 0; };     // gnu-warning {{struct without named members is a GNU extension}}
union U {};                // expected-warning {{empty union is a GNU extension}} \
                           // gnu-warning {{empty union is a GNU extension}}

void compound_literal_empty_record(void) {
  (void)(struct {}){};     // expected-warning {{empty struct is a GNU extension}} \
                           // gnu-warning {{empty struct is a GNU extension}} \
                           // gnu-warning {{use of an empty initializer is a C23 extension}}
}
