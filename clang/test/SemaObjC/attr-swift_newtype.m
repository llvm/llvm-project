// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int Bad1 __attribute__((swift_newtype(invalid)));
// expected-warning@-1 {{'swift_newtype' attribute argument not supported: 'invalid'}}
typedef int Bad2 __attribute__((swift_newtype()));
// expected-error@-1 {{argument required after attribute}}
typedef int Bad3 __attribute__((swift_newtype(invalid, ignored)));
// expected-error@-1 {{expected ')'}}
// expected-note@-2 {{to match this '('}}
// expected-warning@-3 {{'swift_newtype' attribute argument not supported: 'invalid'}}

struct __attribute__((__swift_newtype__(struct))) Bad4 {};
// expected-error@-1 {{'__swift_newtype__' attribute only applies to typedefs}}
