// RUN: %clang_cc1 -fsyntax-only -triple=x86_64-linux -verify %s

// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
static_assert(true, ""1);

// expected-error@+1 {{unexpected ';' before ')'}}
static_assert(true, "";);

// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
static_assert(true, ""
