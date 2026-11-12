// RUN: %clang_cc1 -std=c++11 -verify %s

const char *str1 = R"(abcdef)"; // ok
const char *str2 = R"foo()foo"; // ok
const char *str3 = R"()"; // ok
// FIXME: recover better than this.
const char *str4 = R"abcdefghijkmnopqrstuvwxyz(abcdef)abcdefghijkmnopqrstuvwxyz"; // expected-error {{raw string delimiter longer than 16 characters}} expected-error {{expected expression}}
