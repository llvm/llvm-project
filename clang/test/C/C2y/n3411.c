// RUN: %clang_cc1 -verify=good -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify -Wnewline-eof -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=good -std=c23 -Wall -pedantic %s
// RUN: %clang_cc1 -verify=good -std=c23 %s
// RUN: %clang_cc1 -verify -Wnewline-eof -std=c23 %s

/* WG14 N3411: Yes
 * Slay Some Earthly Demons XII
 *
 * Allow a non-empty source file to end without a final newline character. Note
 * that this file intentionally does not end with a trailing newline.
 */
// good-no-diagnostics

int x; // Ensure the file contains at least one declaration.
// expected-warning {{no newline at end of file}}