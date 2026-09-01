// RUN: %clang_cc1 -E -verify %s

#pragma clang __set_pp_state // expected-error {{expected identifier after '#pragma clang __set_pp_state'}}
#pragma clang __set_pp_state 123984 // expected-error {{expected identifier after '#pragma clang __set_pp_state'}}

#pragma clang __set_pp_state foo // expected-error {{invalid argument 'foo' in '#pragma clang __set_pp_state'}}
#pragma clang __set_pp_state void // expected-error {{invalid argument 'void' in '#pragma clang __set_pp_state'}}

#pragma clang __set_pp_state __GLIBCXX__ foo // expected-error {{expected integer after '#pragma clang __set_pp_state __GLIBCXX__'}}
#pragma clang __set_pp_state __GLIBCXX__ 100000000000000000000000000000 // expected-error {{expected integer after '#pragma clang __set_pp_state __GLIBCXX__'}}
#pragma clang __set_pp_state __GLIBCXX__ 42.0 // expected-error {{expected integer after '#pragma clang __set_pp_state __GLIBCXX__'}}

#pragma clang __set_pp_state __GLIBCXX__ 42L
#pragma clang __set_pp_state __GLIBCXX__ 42

// Check that we treat the identifier after '__set_pp_state' literally.
#define MACRO __GLIBCXX__
#pragma clang __set_pp_state MACRO // expected-error {{invalid argument 'MACRO' in '#pragma clang __set_pp_state'}}

// Check that we treat '__set_pp_state' literally.
#define __set_pp_state foobar
#pragma clang __set_pp_state __GLIBCXX__ 42

// The pragma does *not* define __GLIBCXX__!
#ifdef __GLIBCXX__
#   error __set_pp_state __GLIBCXX__ should not define __GLIBCXX__
#endif
