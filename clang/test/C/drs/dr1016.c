/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify=c89only -pedantic %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c23 -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c2y -fsyntax-only -verify -pedantic %s
 */

/* expected-no-diagnostics */

/* WG14 DR1016: yes
 * Are digit separators allowed in __LINE__?
 *
 * We cannot exhaustively test that digit separators do not appear in __LINE__,
 * but if any line number were to have digit separators, it would be one of the
 * largest line numbers we support. So we'll test that it does not produce any
 * digit separators and call it good. If a digit separator is produced by the
 * expansion, then the variable declaration would be invalid because it would
 * contain single-quote characters.
 */
#define UNIQ_(X, Y) X ## Y
#define UNIQ(X) UNIQ_(X, __LINE__)
#line 2147483646       /* c89only-warning {{C requires #line number to be less than 32768, allowed as extension}} */
int UNIQ(special) = 0; /* okay */

