// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic -Wno-typedef-redefinition -std=c99

// Make sure we accept a single typedef
typedef int (*a)[!.0]; // expected-warning{{folded to constant array}}

// And make sure we accept identical redefinitions in system headers
// (The test uses -Wno-typedef-redefinition to simulate this.)
typedef int (*a)[!.0]; // expected-warning{{folded to constant array}}
