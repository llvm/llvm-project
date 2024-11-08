// RUN: %clang_cc1 -x c -std=c2x -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c -std=c2x -E -DPP_ONLY=1 %s | FileCheck %s --strict-whitespace

/* WG14 N2836: Clang 15
 *   Identifier Syntax using Unicode Standard Annex 31
 */

/* WG14 N2939: Clang 15
 *   Identifier Syntax Fixes
 */

// Some of the tests below are derived from clang/test/Lexer/unicode.c.

// This file contains Unicode characters; please do not "fix" them!

// No diagnostics for pragma directives.
#pragma mark ¡Unicode!

// lone non-identifier characters are allowed in preprocessing.
#define COPYRIGHT Copyright © 2012
#define XSTR(X) #X
#define STR(X) XSTR(X)

static const char *copyright = STR(COPYRIGHT); // no-warning
// CHECK: static const char *copyright = "Copyright © {{2012}}";

#if PP_ONLY
COPYRIGHT
// CHECK: Copyright © {{2012}}
#endif

// The characters in the following identifiers are no longer valid as either
// start or continuation characters as of C23. These are taken from section 1
// of N2836.
extern int \N{CONSTRUCTION WORKER};  // expected-error {{expected identifier or '('}}
extern int X\N{CONSTRUCTION WORKER}; // expected-error {{character <U+1F477> not allowed in an identifier}}
extern int \U0001F477;  // expected-error {{expected identifier or '('}}
extern int X\U0001F477; // expected-error {{character <U+1F477> not allowed in an identifier}}
extern int 👷;  // expected-error {{unexpected character <U+1F477>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X👷; // expected-error {{character <U+1F477> not allowed in an identifier}}
extern int 🕐;  // expected-error {{unexpected character <U+1F550>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X🕐; // expected-error {{character <U+1F550> not allowed in an identifier}}
extern int 💀;  // expected-error {{unexpected character <U+1F480>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X💀; // expected-error {{character <U+1F480> not allowed in an identifier}}
extern int 👊;  // expected-error {{unexpected character <U+1F44A>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X👊; // expected-error {{character <U+1F44A> not allowed in an identifier}}
extern int 🚀;  // expected-error {{unexpected character <U+1F680>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X🚀; // expected-error {{character <U+1F680> not allowed in an identifier}}
extern int 😀;  // expected-error {{unexpected character <U+1F600>}} \
                // expected-warning {{declaration does not declare anything}}
extern int X😀; // expected-error {{character <U+1F600> not allowed in an identifier}}

// The characters in the following identifiers are not allowed as start
// characters, but are allowed as continuation characters.
extern int \N{ARABIC-INDIC DIGIT ZERO}; // expected-error {{expected identifier or '('}}
extern int X\N{ARABIC-INDIC DIGIT ZERO};
extern int \u0661; // expected-error {{expected identifier or '('}}
extern int X\u0661;
extern int ٢;  // expected-error {{character <U+0662> not allowed at the start of an identifier}} \\
               // expected-warning {{declaration does not declare anything}}
extern int X٠;

// The characters in the following identifiers are not valid start or
// continuation characters in the standard, but are accepted as a conforming
// extension.
extern int \N{SUPERSCRIPT ZERO};  // expected-error {{expected identifier or '('}}
extern int X\N{SUPERSCRIPT ZERO}; // expected-warning {{mathematical notation character <U+2070> in an identifier is a Clang extension}}
extern int \u00B9;  // expected-error {{expected identifier or '('}}
extern int X\u00B9; // expected-warning {{mathematical notation character <U+00B9> in an identifier is a Clang extension}}
extern int ²;  // expected-error {{character <U+00B2> not allowed at the start of an identifier}} \\
               // expected-warning {{declaration does not declare anything}}
extern int X²; // expected-warning {{mathematical notation character <U+00B2> in an identifier is a Clang extension}}
extern int \N{PARTIAL DIFFERENTIAL};  // expected-warning {{mathematical notation character <U+2202> in an identifier is a Clang extension}}
extern int X\N{PARTIAL DIFFERENTIAL}; // expected-warning {{mathematical notation character <U+2202> in an identifier is a Clang extension}}
extern int \u2207;  // expected-warning {{mathematical notation character <U+2207> in an identifier is a Clang extension}}
extern int X\u2207; // expected-warning {{mathematical notation character <U+2207> in an identifier is a Clang extension}}
extern int ∞;  // expected-warning {{mathematical notation character <U+221E> in an identifier is a Clang extension}}
extern int X∞; // expected-warning {{mathematical notation character <U+221E> in an identifier is a Clang extension}}
