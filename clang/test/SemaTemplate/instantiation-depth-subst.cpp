// RUN: %clang_cc1 -std=c++11 -verify %s -ftemplate-depth=2

// PR9793
template<typename T> auto f(T t) -> decltype(f(t)); // \
// expected-error {{recursive template instantiation exceeded maximum depth of 2}} \
// expected-note 2 {{while substituting}}

struct S {};
int k = f(S{}); // expected-note {{while substituting}}
