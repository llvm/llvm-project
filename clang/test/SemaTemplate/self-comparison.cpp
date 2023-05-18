// RUN: %clang_cc1 -fsyntax-only -verify %s

template <int A, int B> void foo() {
  (void)(A == A); // expected-warning {{self-comparison always evaluates to true}}
  (void)(A == B);
}
template <int A, int B> struct S1 {
  void foo() {
    (void)(A == A); // expected-warning {{self-comparison always evaluates to true}}
    (void)(A == B);
  }
};

template <int A, int B> struct S2 {
  template <typename T> T foo() {
    (void)(A == A); // expected-warning {{self-comparison always evaluates to true}}
    (void)(A == B);
  }
};

struct S3 {
  template <int A, int B> void foo() {
    (void)(A == A); // expected-warning {{self-comparison always evaluates to true}}
    (void)(A == B);
  }
};

template <int A> struct S4 {
  template <int B> void foo() {
    (void)(A == A); // expected-warning {{self-comparison always evaluates to true}}
    (void)(A == B);
  }
};

const int N = 42;
template <int X> void foo2() {
  (void)(X == N);
  (void)(N == X);
}

void test() {
  foo<1, 1>();
  S1<1, 1> s1; s1.foo();
  S2<1, 1> s2; s2.foo<void>();
  S3 s3; s3.foo<1, 1>();
  S4<1> s4; s4.foo<1>();
  foo2<N>();
}
