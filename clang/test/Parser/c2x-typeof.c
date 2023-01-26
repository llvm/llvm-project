// RUN: %clang_cc1 -verify -std=c2x %s

// Demonstrate that we don't support the expression form without parentheses in
// C2x mode.
typeof 0 int i = 12;         // expected-error {{expected '(' after 'typeof'}} expected-error {{expected identifier or '('}}
typeof 0 j = 12;             // expected-error {{expected '(' after 'typeof'}} expected-error {{expected identifier or '('}}
typeof_unqual 0 k = 12;      // expected-error {{expected '(' after 'typeof_unqual'}} expected-error {{expected identifier or '('}}
typeof_unqual 0 int l = 12;  // expected-error {{expected '(' after 'typeof_unqual'}} expected-error {{expected identifier or '('}}

// Show that combining typeof with another type specifier fails, but otherwise
// the expression and type forms are both parsed properly.
typeof(0) int a = 12;        // expected-error {{cannot combine with previous 'typeof' declaration specifier}}
typeof(0) b = 12;
typeof_unqual(0) int c = 12; // expected-error {{cannot combine with previous 'typeof_unqual' declaration specifier}}
typeof_unqual(0) d = 12;
typeof(int) e = 12;
typeof_unqual(int) f = 12;

// Show that we can parse nested constructs of both forms.
typeof(typeof(0)) w;
typeof_unqual(typeof(0)) x;
typeof(typeof_unqual(0)) y;
typeof_unqual(typeof_unqual(0)) z;

// Show that you can spell the type in functions, structures, or as the base
// type of an enumeration.
typeof(b) func1(typeof(b) c);
typeof_unqual(b) func2(typeof_unqual(b) c);

struct S {
  typeof(b) i;
  typeof_unqual(b) j;
} s;

enum E1 : typeof(b) { FirstZero };
enum E2 : typeof_unqual(b) { SecondZero };

// Show that you can use this type in place of another type and everything
// works as expected.
_Static_assert(__builtin_offsetof(typeof(struct S), i) == 0);
_Static_assert(__builtin_offsetof(typeof(s), i) == 0);
_Static_assert(__builtin_offsetof(typeof_unqual(struct S), i) == 0);
_Static_assert(__builtin_offsetof(typeof_unqual(s), i) == 0);

