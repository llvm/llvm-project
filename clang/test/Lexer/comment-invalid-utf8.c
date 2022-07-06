// RUN: %clang_cc1 -fsyntax-only %s -Winvalid-utf8 -verify=expected
// RUN: %clang_cc1 -fsyntax-only %s -verify=nowarn
// nowarn-no-diagnostics

// This file is purposefully encoded as windows-1252
// be careful when modifying.

//€
// expected-warning@-1 {{invalid UTF-8 in comment}}

// € ‚ƒ„…†‡ˆ‰ Š ‹ Œ Ž
// expected-warning@-1 6{{invalid UTF-8 in comment}}

/*€*/
// expected-warning@-1 {{invalid UTF-8 in comment}}

/*€ ‚ƒ„…†‡ˆ‰ Š ‹ Œ Ž*/
// expected-warning@-1 6{{invalid UTF-8 in comment}}

/*
€
*/
// expected-warning@-2 {{invalid UTF-8 in comment}}

// abcd
// €abcd
// expected-warning@-1 {{invalid UTF-8 in comment}}


//Â§ Â§ Â§ ðŸ˜€ ä½ å¥½ Â©

/*Â§ Â§ Â§ ðŸ˜€ ä½ å¥½ Â©*/

/*
Â§ Â§ Â§ ðŸ˜€ ä½ å¥½ Â©
*/

/* Â§ Â§ Â§ ðŸ˜€ ä½ å¥½ Â© */
