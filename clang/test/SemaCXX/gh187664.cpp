// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { // #A
};

struct B {
  struct C{}; // #B_C
  enum E{};
};

namespace NS {
  struct D{};
}

enum E {}; // #E

typedef int T;
using U = int;

// Basic sanity check on the original fuzzing generated syntax
template <class T = U struct B::C> void test_template();
// expected-error@-1{{cannot combine with previous 'type-name' declaration specifier}}

// The bug we're looking at is the result of parsing repeated named type
// specifiers. All the record types actually go through a single path, so
// we don't repeat all of them. The first few are a basic sanity check of
// each one anyway
A struct  B::C test01;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
A union   B::C test02;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
// expected-error@-2 {{use of 'C' with tag type that does not match previous declaration}}
// expected-note@#B_C {{previous use is here}}
A class   B::C test03;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
A struct NS::D test04;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
A struct   ::A test05;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
T struct  B::C test06;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
U struct  B::C test07;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}

// enums _do_ need separate testing as they have a different syntax, and so
// take a different path through the parser
E struct  B::C test08;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
A enum     ::A test09;
// expected-error@-1 {{use of 'A' with tag type that does not match previous declaration}}
// expected-note@#A {{previous use is here}}

A enum     ::E test10;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
E enum    B::E test11;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
E enum     ::E test12;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
T enum     ::E test13;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
E struct   ::E test14;
// expected-error@-1 {{use of 'E' with tag type that does not match previous declaration}}
// expected-note@#E {{previous use is here}}

// These cases also _technically_ go wrong, but when the initial type specifier
// is namespaced, we get the required info from the parser directly, which avoids
// the reading from the corrupted scope
B::C struct ::B   test15;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
B::C struct B::C  test16;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
B::C struct NS::D test17;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}

using Test18 = A struct ::A;
// expected-error@-1 {{cannot combine with previous 'type-name' declaration specifier}}
