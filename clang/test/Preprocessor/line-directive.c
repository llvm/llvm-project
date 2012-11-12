// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -E %s 2>&1 | grep 'blonk.c:92:2: error: ABC'
// RUN: %clang_cc1 -E %s 2>&1 | grep 'blonk.c:93:2: error: DEF'

#line 'a'            // expected-error {{#line directive requires a positive integer argument}}
#line 0              // expected-warning {{#line directive with zero argument is a GNU extension}}
#line 00             // expected-warning {{#line directive with zero argument is a GNU extension}}
#line 2147483648     // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
#line 42             // ok
#line 42 'a'         // expected-error {{invalid filename for #line directive}}
#line 42 "foo/bar/baz.h"  // ok


// #line directives expand macros.
#define A 42 "foo"
#line A

# 42
# 42 "foo"
# 42 "foo" 2 // expected-error {{invalid line marker flag '2': cannot pop empty include stack}}
# 42 "foo" 1 3  // enter
# 42 "foo" 2 3  // exit
# 42 "foo" 2 3 4 // expected-error {{invalid line marker flag '2': cannot pop empty include stack}}
# 42 "foo" 3 4

# 'a'            // expected-error {{invalid preprocessing directive}}
# 42 'f'         // expected-error {{invalid filename for line marker directive}}
# 42 1 3         // expected-error {{invalid filename for line marker directive}}
# 42 "foo" 3 1   // expected-error {{invalid flag line marker directive}}
# 42 "foo" 42    // expected-error {{invalid flag line marker directive}}
# 42 "foo" 1 2   // expected-error {{invalid flag line marker directive}}


// These are checked by the RUN line.
#line 92 "blonk.c"
#error ABC
#error DEF
// expected-error@-2 {{ABC}}
#line 150
// expected-error@-3 {{DEF}}


// Verify that linemarker diddling of the system header flag works.

# 192 "glomp.h" // not a system header.
typedef int x;  // expected-note {{previous definition is here}}
typedef int x;  // expected-warning {{redefinition of typedef 'x' is a C11 feature}}

# 192 "glomp.h" 3 // System header.
typedef int y;  // ok
typedef int y;  // ok

typedef int q;  // q is in system header.

#line 42 "blonk.h"  // doesn't change system headerness.

typedef int z;  // ok
typedef int z;  // ok

# 97     // doesn't change system headerness.

typedef int z1;  // ok
typedef int z1;  // ok

# 42 "blonk.h"  // DOES change system headerness.

typedef int w;  // expected-note {{previous definition is here}}
typedef int w;  // expected-warning {{redefinition of typedef 'w' is a C11 feature}}

typedef int q;  // original definition in system header, should not diagnose.

// This should not produce an "extra tokens at end of #line directive" warning,
// because #line is allowed to contain expanded tokens.
#define EMPTY()
#line 2 "foo.c" EMPTY( )
#line 2 "foo.c" NONEMPTY( )  // expected-warning{{extra tokens at end of #line directive}}

// PR3940
#line 0xf  // expected-error {{#line directive requires a simple digit sequence}}
#line 42U  // expected-error {{#line directive requires a simple digit sequence}}


// Line markers are digit strings interpreted as decimal numbers, this is
// 10, not 8.
#line 010  // expected-warning {{#line directive interprets number as decimal, not octal}}
extern int array[__LINE__ == 10 ? 1:-1];

/* PR3917 */
#line 41
extern char array2[\
_\
_LINE__ == 42 ? 1: -1];  /* line marker is location of first _ */

// rdar://11550996
#line 0 "line-directive.c" // expected-warning {{#line directive with zero argument is a GNU extension}}
undefined t; // expected-error {{unknown type name 'undefined'}}


