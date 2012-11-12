// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check that we don't allow illegal uses of inline
// (checking C++-only constructs here)
struct c {inline int a;}; // expected-error{{'inline' can only appear on functions}}
