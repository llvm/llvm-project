// RUN: %clang_cc1 -fsyntax-only -verify %s -DMAX=128 -foperator-arrow-depth=128
// RUN: %clang_cc1 -fsyntax-only -verify %s -DMAX=2 -foperator-arrow-depth=2
// RUN: %clang -fsyntax-only -Xclang -verify %s -DMAX=10 -foperator-arrow-depth=10

template<int N> struct B;
template<int N> struct A {
  B<N> operator->(); // expected-note +{{'operator->' declared here produces an object of type 'B<}}
};
template<int N> struct B {
  A<N-1> operator->(); // expected-note +{{'operator->' declared here produces an object of type 'A<}}
#if MAX != 2
  // expected-note-re@-2 {{(skipping {{120|2}} 'operator->'s in backtrace)}}
#endif
};

struct X { int n; };
template<> struct B<1> {
  X *operator->();
};

A<MAX/2> good;
int n = good->n;

B<MAX/2 + 1> bad;
int m = bad->n; // expected-error-re {{use of 'operator->' on type 'B<{{2|10|128}} / 2 + 1>' would invoke a sequence of more than {{2|10|128}} 'operator->' calls}}
                // expected-note@-1 {{use -foperator-arrow-depth=N to increase 'operator->' limit}}
