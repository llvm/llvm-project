// RUN: %clang_cc1 -fsyntax-only -verify -ftemplate-backtrace-limit 2 %s

template<int N, typename T> struct X : X<N+1, T*> {};
// expected-error-re@3 {{recursive template instantiation exceeded maximum depth of 1024{{$}}}}
// expected-note@3 {{instantiation of template class}}
// expected-note@3 {{skipping 1023 contexts in backtrace}}
// expected-note@3 {{use -ftemplate-depth=N to increase recursive template instantiation depth}}

X<0, int> x; // expected-note {{in instantiation of}}
