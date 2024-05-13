// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-field-initializers %s

// This was PR4808.

struct Foo { int a, b; };

struct Foo foo0 = { 1 }; // expected-warning {{missing field 'b' initializer}}
struct Foo foo1 = { .a = 1 }; // designator avoids MFI warning
struct Foo foo2 = { .b = 1 }; // designator avoids MFI warning

struct Foo bar0[] = {
  { 1,2 },
  { 1 },   // expected-warning {{missing field 'b' initializer}}
  { 1,2 }
};

struct Foo bar1[] = {
  1, 2,
  1, 2,
  1
}; // expected-warning {{missing field 'b' initializer}}

struct Foo bar2[] = { {}, {}, {} };

struct One { int a; int b; };
struct Two { float c; float d; float e; };

struct Three {
    union {
        struct One one;
        struct Two two;
    } both;
};

struct Three t0 = {
    { .one = { 1, 2 } }
};
struct Three t1 = {
    { .two = { 1.0f, 2.0f, 3.0f } }
};

struct Three data[] = {
  { { .one = { 1, 2 } } },
  { { .one = { 1 } } }, // expected-warning {{missing field 'b' initializer}}
  { { .two = { 1.0f, 2.0f, 3.0f } } },
  { { .two = { 1.0f, 2.0f } } } // expected-warning {{missing field 'e' initializer}}
};

struct { int:5; int a; int:5; int b; int:5; } noNamedImplicit[] = {
  { 1, 2 },
  { 1 } // expected-warning {{missing field 'b' initializer}}
};

// GH66300
struct S {
  int f0;
  int f1[];
};

// We previously would accidentally diagnose missing a field initializer for
// f1, now we no longer issue that warning (note, this code is still unsafe
// because of the buffer overrun).
struct S s = {1, {1, 2}};
