// RUN: %clang_cc1 -verify -fsyntax-only %s

typedef int T1 __attribute__((swift_newtype(struct)));
typedef int T2 __attribute__((swift_newtype(enum)));

typedef int T3 __attribute__((swift_wrapper(struct)));
typedef int T4 __attribute__((swift_wrapper(enum)));


typedef int Bad1 __attribute__((swift_newtype(bad))); // expected-warning{{'swift_newtype' attribute argument not supported: 'bad'}}
typedef int Bad2 __attribute__((swift_newtype())); // expected-error{{argument required after attribute}}
typedef int Bad3 __attribute__((swift_newtype(bad, badder)));
  // expected-error@-1{{expected ')'}}
  // expected-note@-2{{to match this '('}}
  // expected-warning@-3{{'swift_newtype' attribute argument not supported: 'bad'}}


// TODO: better error message below
// FIXME: why is this a parse error, rather than Sema error triggering?
struct Bad4 __attribute__((swift_newtype(struct))) { }; // expected-error{{expected identifier or '('}}
