// RUN: %clang_cc1 -fsyntax-only -Wunused -std=c++1z -verify %s

struct [[maybe_unused]] S1 {}; // ok
struct [[maybe_unused maybe_unused]] S2 {}; // expected-error {{attribute 'maybe_unused' cannot appear multiple times in an attribute specifier}}
struct [[maybe_unused("Wrong")]] S3 {}; // expected-error {{'maybe_unused' cannot have an argument list}}
