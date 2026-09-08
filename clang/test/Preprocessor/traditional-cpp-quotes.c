/* RUN: %clang_cc1 -traditional-cpp -E -verify %s
 * expected-no-diagnostics
 */

/* -traditional-cpp is mainly used to preprocess non-source files, where an
 * unbalanced quote is ordinary text rather than the start of a literal.
 * GCC does not diagnose these either. */
// Test "double
// Try 'single
he said '' and left
don't
