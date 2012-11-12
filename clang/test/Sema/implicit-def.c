/* RUN: %clang_cc1 -fsyntax-only %s -std=c89
 * RUN: not %clang_cc1 -fsyntax-only %s -std=c99 -pedantic-errors
 */

int A() {
  return X();
}

