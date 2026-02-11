// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s

/* WG14 N3347: Yes
 * Slay Some Earthly Demons IX
 *
 * Declarations of a tentative definition with internal linkage must be
 * complete by the end of the translation unit.
 */

struct foo; // #foo
static struct foo f1; /* expected-warning {{tentative definition of variable with internal linkage has incomplete non-array type 'struct foo'}}
                         expected-error {{tentative definition has type 'struct foo' that is never completed}}
                         expected-note@#foo 2 {{forward declaration of 'struct foo'}}
                       */

extern struct foo f2; // Ok, does not have internal linkage

struct bar; // #bar
static struct bar b; /* expected-warning {{tentative definition of variable with internal linkage has incomplete non-array type 'struct bar'}}
                        expected-note@#bar {{forward declaration of 'struct bar'}}
                      */
struct bar { int x; };

