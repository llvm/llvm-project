// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify -pedantic %s
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'blonk.c:98:2: error: ABC'
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'blonk.c:99:2: error: DEF'
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'line-directive.c:26:2: error: MAIN7'
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'enter-1:244:2: error: ENTER1'
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'main:250:2: error: MAIN2'

// expected-error@+1{{cannot pop empty include stack}}
# 18 "" 2

// a push/pop before any other line control
# 24 "enter-0" 1 // expected-warning {{this style of line directive is a GNU extension}}
# 26 "" 2 // pop to main file: expected-warning {{this style of line directive is a GNU extension}}
#error MAIN7
// expected-error@-1{{MAIN7}}

#line 'a'            // expected-error {{#line directive requires a positive integer argument}}
#line 0              // expected-warning {{#line directive with zero argument is a GNU extension}}
#line 00             // expected-warning {{#line directive with zero argument is a GNU extension}}
#line 2147483648     // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
#line 42             // ok
#line 44 'a'         // expected-error {{invalid filename for #line directive}}
#line 46 "foo/bar/baz.h"  // ok


// #line directives expand macros.
#define A 54 "foo"
#line A

# 60 // expected-warning {{this style of line directive is a GNU extension}}
# 62 "foo" // expected-warning {{this style of line directive is a GNU extension}}
# 64 "foo" 2 // expected-error {{invalid line marker flag '2': cannot pop empty include stack}}
// The next two lines do not get diagnosed because they are considered to be
// within the system header, where diagnostics are suppressed.
# 70 "foo" 1 3  // enter expected-warning {{this style of line directive is a GNU extension}}
# 72 "foo" 2 3  // exit
# 74 "foo" 2 3 4 // expected-error {{invalid line marker flag '2': cannot pop empty include stack}}
# 76 "foo" 3 4

# 'a'            // expected-error {{invalid preprocessing directive}}
# 82 'f'         // expected-error {{invalid filename for line marker directive}}
# 84 1 3         // expected-error {{invalid filename for line marker directive}}
# 86 "foo" 3 1   // expected-error {{invalid flag line marker directive}}
# 88 "foo" 42    // expected-error {{invalid flag line marker directive}}
# 90 "foo" 1 2   // expected-error {{invalid flag line marker directive}}
# 92a33          // expected-error {{GNU line marker directive requires a simple digit sequence}}

// These are checked by the RUN line.
#line 98 "blonk.c"
#error ABC
#error DEF
// expected-error@-2 {{ABC}}
#line 106
// expected-error@-3 {{DEF}}


// Verify that linemarker diddling of the system header flag works.
# 116 "" 3 4
# 118 "glomp.h" // in a system header.
typedef int x;  // expected-note {{previous definition is here}}
typedef int x;  // expected-warning {{redefinition of typedef 'x' is a C11 feature}}

# 126 "glomp.h" 3 // not in a system header. expected-warning {{this style of line directive is a GNU extension}}
typedef int y;  // ok
typedef int y;  // ok

typedef int q;  // q is in system header.

#line 138 "blonk.h"  // doesn't change system headerness.

typedef int z;  // ok
typedef int z;  // ok

# 148     // doesn't change system headerness.

typedef int z1;  // ok
typedef int z1;  // ok

# 158 "blonk.h"  // DOES change system headerness.

typedef int w;  // expected-note {{previous definition is here}}
typedef int w;  // expected-warning {{redefinition of typedef 'w' is a C11 feature}}

typedef int q;  // original definition in system header, should not diagnose.

// This should not produce an "extra tokens at end of #line directive" warning,
// because #line is allowed to contain expanded tokens.
#define EMPTY()
#line 178 "foo.c" EMPTY( )
#line 180 "foo.c" NONEMPTY( )  // expected-warning{{extra tokens at end of #line directive}}

// PR3940
#line 0xf  // expected-error {{#line directive requires a simple digit sequence}}
#line 198U  // expected-error {{#line directive requires a simple digit sequence}}


// Line markers are digit strings interpreted as decimal numbers, this is
// 10, not 8.
#line 010  // expected-warning {{#line directive interprets number as decimal, not octal}}
extern int array[__LINE__ == 10 ? 1:-1];

# 020      // expected-warning {{GNU line marker directive interprets number as decimal, not octal}} expected-warning {{this style of line directive is a GNU extension}}
extern int array_gnuline[__LINE__ == 20 ? 1:-1];

/* PR3917 */
#line 41
extern char array2[\
_\
_LINE__ == 42 ? 1: -1];  /* line marker is location of first _ */

# 51 // expected-warning {{this style of line directive is a GNU extension}}
extern char array2_gnuline[\
_\
_LINE__ == 52 ? 1: -1];  /* line marker is location of first _ */

#line 0 "line-directive.c" // expected-warning {{#line directive with zero argument is a GNU extension}}
undefined t; // expected-error {{unknown type name 'undefined'}}

# 238 "main" // expected-warning {{this style of line directive is a GNU extension}}
# 240 "enter-1" 1 // expected-warning {{this style of line directive is a GNU extension}}
# 242 "enter-2" 1 // expected-warning {{this style of line directive is a GNU extension}}
# 244 "" 2 // pop to enter-1: expected-warning {{this style of line directive is a GNU extension}}
#error ENTER1
// expected-error@-1{{ENTER1}}
# 250 "" 2 // pop to "main": expected-warning {{this style of line directive is a GNU extension}}
#error MAIN2
// expected-error@-1{{MAIN2}}

#line 258 L"wide" // expected-error {{invalid filename for #line directive}}
#line 260 "\x12"
# 262 U"hello" // expected-error {{invalid filename for line marker directive}}
# 264 "\x13" // expected-warning {{this style of line directive is a GNU extension}}

#line 273
// #UNTERMINATED
#line 274 "foo\""
// expected-warning@#UNTERMINATED {{missing terminating '"' character}}
// expected-warning@#UNTERMINATED {{extra tokens at end of #line directive}}

#line 280 <> // expected-error {{invalid filename for #line directive}}
#line 282 <foo> // expected-error {{invalid filename for #line directive}}
