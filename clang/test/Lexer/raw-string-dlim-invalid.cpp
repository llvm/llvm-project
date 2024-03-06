// RUN: %clang_cc1 -E -fsyntax-only -verify %s

// expected-error@+2{{invalid character ')' in raw string delimiter; use PREFIX( )PREFIX to delimit raw string}}
// expected-error@+1{{expected expression}}
char const *str1 = R")";

// expected-error@+2{{invalid newline character in raw string delimiter; use PREFIX( )PREFIX to delimit raw string}}
// expected-error@+1{{expected expression}}
char const* str2 = R"";
