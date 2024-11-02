/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -DC89 -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -verify -DC99 -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -DC11 %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -verify -DC17 %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -verify -DC2X %s
 */

/* expected-no-diagnostics */

/* WG14 DR411: yes
 * Predefined macro values
 *
 * Note: the DR is about the C11 macro value, but we'll test all the standard
 * version macro values just to be sure. We do not need to test
 * __STDC_LIB_EXT1__ values because that requires an Annex K-compatible header.
 */
#if defined(C89)
#ifdef __STDC_VERSION__
#error "C89 didn't have this macro!"
#endif
#elif defined(C99)
_Static_assert(__STDC_VERSION__ == 199901L, "");
#elif defined(C11)
_Static_assert(__STDC_VERSION__ == 201112L, "");
#elif defined(C17)
_Static_assert(__STDC_VERSION__ == 201710L, "");
#elif defined(C2X)
/* FIXME: this value will change once WG14 picks the final value for C2x. */
_Static_assert(__STDC_VERSION__ == 202000L, "");
#else
#error "unknown language standard version"
#endif

