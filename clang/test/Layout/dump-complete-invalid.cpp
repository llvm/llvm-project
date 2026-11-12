// RUN: %clang_cc1 -verify -fsyntax-only -fdump-record-layouts-complete %s

struct Incomplete; // expected-note {{forward declaration}}

// Check we don't crash on trying to print out an invalid declaration.
struct Invalid : Incomplete {}; // expected-error {{base class has incomplete type}}
