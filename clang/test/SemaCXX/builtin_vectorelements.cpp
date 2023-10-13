// RUN: %clang_cc1 -triple aarch64 -std=c++17 -fsyntax-only -verify %s

template <typename T>
using VecT __attribute__((vector_size(16))) = T;

struct FooT {
  template <typename T>
  using VecT __attribute__((vector_size(8))) = T;
};

void test_builtin_vectorelements() {
  using veci4 __attribute__((vector_size(16))) = int;
  (void) __builtin_vectorelements(veci4);

  using some_other_vec = veci4;
  (void) __builtin_vectorelements(some_other_vec);

  using some_int = int;
  (void) __builtin_vectorelements(some_int); // expected-error {{'__builtin_vectorelements' argument must be a vector}}

  class Foo {};
  __builtin_vectorelements(Foo); // expected-error {{'__builtin_vectorelements' argument must be a vector}}

  struct Bar { veci4 vec; };
  (void) __builtin_vectorelements(Bar{}.vec);

  struct Baz { using VecT = veci4; };
  (void) __builtin_vectorelements(Baz::VecT);

  (void) __builtin_vectorelements(FooT::VecT<long>);
  (void) __builtin_vectorelements(VecT<char>);
}

