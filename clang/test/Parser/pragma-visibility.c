// RUN: %clang_cc1 -fsyntax-only -verify %s

#pragma GCC visibility foo // expected-warning{{expected identifier in '#pragma visibility' - ignored}}
#pragma GCC visibility pop foo // expected-warning{{extra tokens at end of '#pragma visibility' - ignored}}
#pragma GCC visibility push // expected-warning{{missing '(' after '#pragma visibility'}}
#pragma GCC visibility push( // expected-warning{{expected identifier in '#pragma visibility' - ignored}}
#pragma GCC visibility push(hidden // expected-warning{{missing ')' after '#pragma visibility' - ignoring}}
#pragma GCC visibility push(hidden)
#pragma GCC visibility pop
