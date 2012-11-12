// RUN: %clang_cc1 -fsyntax-only -verify -ftemplate-depth 128 -ftemplate-backtrace-limit 4 %s

template<int N> struct S {
  typedef typename S<N-1>::type type;
  static int f(int n = S<N-1>::f()); // \
// expected-error{{recursive template instantiation exceeded maximum depth of 128}} \
// expected-note 3 {{instantiation of default function argument}} \
// expected-note {{skipping 125 contexts in backtrace}} \
// expected-note {{use -ftemplate-depth=N to increase recursive template instantiation depth}}

};
template<> struct S<0> {
  typedef int type;
};

// Incrementally instantiate up to S<2048>.
template struct S<128>;
template struct S<256>;
template struct S<384>;
template struct S<512>;
template struct S<640>;
template struct S<768>;
template struct S<896>;
template struct S<1024>;
template struct S<1152>;
template struct S<1280>;
template struct S<1408>;
template struct S<1536>;
template struct S<1664>;
template struct S<1792>;
template struct S<1920>;
template struct S<2048>;

// Check that we actually bail out when we hit the instantiation depth limit for
// the default arguments.
void g() { S<2048>::f(); } // expected-note {{required here}}
