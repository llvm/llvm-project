// RUN: %clang_cc1 -verify=c2y -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -Wnewline-eof -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic %s

/* WG14 N3411: Yes
 * Slay Some Earthly Demons XII
 *
 * Allow a non-empty source file to end without a final newline character. Note
 * that this file intentionally does not end with a trailing newline.
 */
// c2y-no-diagnostics

int x; // Ensure the file contains at least one declaration.
// expected-warning {{no newline at end of file}}