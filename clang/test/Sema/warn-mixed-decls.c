/* RUN: %clang_cc1 -fsyntax-only -verify -std=c89 -pedantic %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify -std=c89 -Wdeclaration-after-statement %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify -std=c99 -Wdeclaration-after-statement %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Wdeclaration-after-statement %s
 */

/* Should not emit diagnostic when not pedantic, not enabled or in C++ Code*/
/* RUN: %clang_cc1 -fsyntax-only -verify=none -std=c89 %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -std=c99 %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -std=c89 -Wall %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -std=c99 -Wall -pedantic %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -std=c11 -Wall -pedantic %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -x c++ %s
 */
/* RUN: %clang_cc1 -fsyntax-only -verify=none -x c++ -Wdeclaration-after-statement %s
 */

/* none-no-diagnostics */

int foo(int i)
{
  i += 1;
  int f = i;
#if __STDC_VERSION__ < 199901L
  /* expected-warning@-2 {{mixing declarations and code is a C99 extension}}*/
#else
  /* expected-warning@-4 {{mixing declarations and code is incompatible with standards before C99}}*/
#endif
  return f;
}
