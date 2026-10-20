// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-keyword %s
// expected-no-diagnostics

// Show that we do not diagnose any C++ keywords in Objective-C.

@class Foo; // Okay, Objective-C @ keyword, not a regular identifier

// FIXME: it would be nice to diagnose this, but it is intentionally allowed
// due to @ and selectors allowing C++ keywords in ways that are supposed to be
// contextually compatible with C++.
int class = 12;

