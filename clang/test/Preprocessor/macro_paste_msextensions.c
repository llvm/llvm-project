// RUN: %clang_cc1 -verify -fms-extensions -Wmicrosoft %s
// RUN: not %clang_cc1 -P -E -fms-extensions %s | FileCheck -strict-whitespace %s

// This horrible stuff should preprocess into (other than whitespace):
//   int foo;
//   int bar;
//   int baz;

int foo;

// CHECK: int foo;

#define comment /##/  dead tokens live here
// expected-warning@+1 {{pasting two '/' tokens}}
comment This is stupidity

int bar;

// CHECK: int bar;

#define nested(x) int x comment cute little dead tokens...

// expected-warning@+1 {{pasting two '/' tokens}}
nested(baz)  rise of the dead tokens

;

// CHECK: int baz
// CHECK: ;


// VC++ allows invalid token pastes: (##baz
#define foo(x) abc(x)
#define bar(y) foo(##baz(y))
bar(q) // expected-error {{type specifier missing}} expected-error {{invalid preprocessing token}} expected-error {{parameter list without types}}

// CHECK: abc(baz(q))


#define str(x) #x
#define collapse_spaces(a, b, c, d) str(a ## - ## b ## - ## c ## d)
collapse_spaces(1a, b2, 3c, d4) // expected-error 4 {{invalid preprocessing token}} expected-error {{expected function body}}

// CHECK: "1a-b2-3cd4"
