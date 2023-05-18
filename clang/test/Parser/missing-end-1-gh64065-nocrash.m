// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Roo // expected-note {{class started here}}
// expected-error@+1 {{missing '@end'}}
@interface // expected-error {{expected identifier}}
