// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-error@+2 {{expected ']'}}
// expected-error@+1 {{expected unqualified-id}}
[[fallthrough;]]
