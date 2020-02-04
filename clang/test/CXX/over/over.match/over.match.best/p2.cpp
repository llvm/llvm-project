// RUN: %clang_cc1 -std=c++2a -verify %s

namespace PR44761 {
  template<typename T> concept X = (sizeof(T) == sizeof(T));

  template<typename T> struct A {
    bool operator<(const A&) const & requires X<T>; // #1
    int operator<=>(const A&) const & requires X<T> && X<int> = delete; // #2
  };
  bool k1 = A<int>() < A<int>(); // not ordered by constraints: prefer non-rewritten form
  bool k2 = A<float>() < A<float>(); // prefer more-constrained 'operator<=>'
  // expected-error@-1 {{deleted}}
  // expected-note@#1 {{candidate}}
  // expected-note@#2 {{candidate function has been explicitly deleted}}
  // expected-note@#2 {{candidate function (with reversed parameter order) has been explicitly deleted}}
}
