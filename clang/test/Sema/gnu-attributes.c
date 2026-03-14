// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct s {};

// FIXME: should warn that declaration attribute in type position is
// being applied to the declaration instead?
struct s __attribute__((used)) foo;

// FIXME: Should warn that type attribute in declaration position is
// being applied to the type instead?
struct s  *bar __attribute__((address_space(1)));

// Should not warn because type attribute is in type position.
struct s *__attribute__((address_space(1))) baz;

// Should not warn because declaration attribute is in declaration position.
struct s *quux __attribute__((used));
