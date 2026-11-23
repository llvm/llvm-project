// RUN: %clang_cc1 -verify -std=c2y %s
// RUN: %clang_cc1 -verify -std=c23 %s

/* WG14 N3478: Yes
 * Slay Some Earthly Demons XIII
 *
 * It was previously UB to end a source file with a partial preprocessing token
 * or a partial comment. Clang has always diagnosed these.
 */

// expected-error@+1 {{unterminated /* comment}}
/*