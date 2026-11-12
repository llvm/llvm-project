/* RUN: %clang_cc1 -x c   -std=c89   -fsyntax-only -verify -pedantic-errors -Wno-empty-translation-unit %s
 * RUN: %clang_cc1 -x c   -std=c99   -fsyntax-only -verify -pedantic-errors -Wno-empty-translation-unit %s
 * RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify -pedantic-errors -Wno-empty-translation-unit %s
 * RUN: %clang_cc1 -x c++ -std=c++11 -fsyntax-only -verify -Wc++98-compat-pedantic -Wno-empty-translation-unit %s
 */

#if !defined(__cplusplus)
#  if __STDC_VERSION__ < 199901L
/* expected-error@21 {{'long long' is an extension when C99 mode is not enabled}} */
#  else
/* expected-no-diagnostics */
#  endif
#else
#  if __cplusplus < 201103L
/* expected-error@21 {{'long long' is a C++11 extension}} */
#  else
/* expected-warning@21 {{'long long' is incompatible with C++98}} */
#  endif
#endif

#if 1 > 2LL
#  error should not happen
#endif

