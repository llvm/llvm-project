// RUN:  %clang_cc1 -std=c++2a -fconcepts-ts -verify %s

template<typename T> concept C = true;
static_assert(C<int>); // expected-error{{sorry, unimplemented concepts feature concept specialization used}}
