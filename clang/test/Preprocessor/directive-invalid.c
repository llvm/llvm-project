// RUN: %clang_cc1 -E -verify %s
// rdar://7683173

#define r_paren )
#if defined( x r_paren  // expected-error {{missing ')' after 'defined'}} \
                        // expected-note {{to match this '('}}
#endif
