// RUN: %clang_cc1 -fsyntax-only -Wall -verify -std=c++11 %s

// Regression test for #50794.

// expected-note@+1 2 {{previous definition is here}}
inline namespace X {}

namespace X {} // expected-warning {{inline namespace reopened as a non-inline namespace}}
namespace X {} // expected-warning {{inline namespace reopened as a non-inline namespace}}
