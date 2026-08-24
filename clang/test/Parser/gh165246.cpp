// RUN: %clang_cc1 -verify %s -std=c++11 -fsyntax-only

int decltype {}
// expected-error@-1 {{expected '(' after 'decltype'}} \
// expected-error@-1 {{expected unqualified-id}}