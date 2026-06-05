// RUN: %clang_cc1 -std=c++17 -O1 -emit-obj -Wno-ignored-attributes -o /dev/null %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// Regression test for issue #197206

class A {};

template <typename B> class C {
  // expected-warning@+1 {{'returns_nonnull' attribute only applies to return values that are pointers}}
  friend B D(C) __attribute__((returns_nonnull)) {
    B E;
    return E;
  }
};

template <typename>
C<A> F();

C<A> H = F<int>(); // expected-note {{in instantiation of template class 'C<A>' requested here}}

void G() { D(H); }
