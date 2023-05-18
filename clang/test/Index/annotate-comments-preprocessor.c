// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs %s

// As long as none of this crashes, we don't care about comments in
// preprocessor directives.

#include "annotate-comments-preprocessor.h" /* Aaa. */ /* Bbb. */
#include "annotate-comments-preprocessor.h" /* Aaa. */
#include "annotate-comments-preprocessor.h" /** Aaa. */
#include "annotate-comments-preprocessor.h" /**< Aaa. */
#include "annotate-comments-preprocessor.h" // Aaa.
#include "annotate-comments-preprocessor.h" /// Aaa.
#include "annotate-comments-preprocessor.h" ///< Aaa.

#define A0 0
#define A1 1 /* Aaa. */
#define A2 1 /** Aaa. */
#define A3 1 /**< Aaa. */
#define A4 1 // Aaa.
#define A5 1 /// Aaa.
#define A6 1 ///< Aaa.

int A[] = { A0, A1, A2, A3, A4, A5, A6 };

#if A0 /** Aaa. */
int f(int a1[A1], int a2[A2], int a3[A3], int a4[A4], int a5[A5], int a6[A6]);
#endif /** Aaa. */

#if A1 /** Aaa. */
int g(int a1[A1], int a2[A2], int a3[A3], int a4[A4], int a5[A5], int a6[A6]);
#endif /* Aaa. */

#pragma once /** Aaa. */

#define FOO      \
  do {           \
    /* Aaa. */   \
    /** Aaa. */  \
    /**< Aaa. */ \
    ;            \
  } while(0)

void h(void) {
  FOO;
}

