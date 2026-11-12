// RUN: %clang_cc1 -verify -std=c2x %s

/* WG14 N2930: yes
 * Consider renaming remove_quals
 */

int remove_quals;
int typeof_unqual; // expected-error {{expected '(' after 'typeof_unqual'}}
typeof_unqual(remove_quals) val;
