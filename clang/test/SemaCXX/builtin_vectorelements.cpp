// RUN: %clang_cc1 -triple x86_64                       -std=c++20 -fsyntax-only -verify -disable-llvm-passes %s
// RUN: %clang_cc1 -triple x86_64                       -std=c++20 -fsyntax-only -verify -disable-llvm-passes -fexperimental-new-constant-interpreter %s

// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -std=c++20 -fsyntax-only -verify -disable-llvm-passes %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -std=c++20 -fsyntax-only -verify -disable-llvm-passes -fexperimental-new-constant-interpreter %s

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
  (void) __builtin_vectorelements(some_int); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

  class Foo {};
  __builtin_vectorelements(Foo); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

  struct Bar { veci4 vec; };
  (void) __builtin_vectorelements(Bar{}.vec);

  struct Baz { using VecT = veci4; };
  (void) __builtin_vectorelements(Baz::VecT);

  (void) __builtin_vectorelements(FooT::VecT<long>);
  (void) __builtin_vectorelements(VecT<char>);

  constexpr int i4 = __builtin_vectorelements(veci4);
  constexpr int i4p8 = __builtin_vectorelements(veci4) + 8;
}


namespace GH216997 {
// Reproducer from GH216997.
using vec __attribute__((vector_size(16))) = int &bar; // expected-error {{type-id cannot have a name}}
int baz = __builtin_vectorelements(vec); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

typedef int veci4 __attribute__((vector_size(16)));
int a = __builtin_vectorelements(veci4 &); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
int b = __builtin_vectorelements(veci4 &&); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
int c = __builtin_vectorelements(const veci4 &); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

veci4 v;
int d = __builtin_vectorelements(decltype((v))); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

template <typename T>
int f() {
  return __builtin_vectorelements(T); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
}
int e = f<veci4>();
int g = f<veci4 &>(); // expected-note {{in instantiation of function template specialization}}

void ok(veci4 &r, veci4 &&rr) {
  (void)__builtin_vectorelements(r);
  (void)__builtin_vectorelements(rr);
  (void)__builtin_vectorelements(const veci4);
}
} // namespace GH216997

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

consteval int consteval_elements() { // expected-error {{consteval function never produces a constant expression}}
  return __builtin_vectorelements(svuint64_t); // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}}  // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}} // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}}
}

void test_bad_constexpr() {
  constexpr int eval = consteval_elements(); // expected-error {{initialized by a constant expression}} // expected-error {{not a constant expression}} // expected-note {{in call}} // expected-note {{in call}}
  constexpr int i32 = __builtin_vectorelements(svuint32_t); // expected-error {{initialized by a constant expression}} // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}}
  constexpr int i16p8 = __builtin_vectorelements(svuint16_t) + 16; // expected-error {{initialized by a constant expression}} // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}}
  constexpr int lambda = [] { return __builtin_vectorelements(svuint16_t); }(); // expected-error {{initialized by a constant expression}} // expected-note {{cannot determine number of elements for sizeless vectors in a constant expression}} // expected-note {{in call}}
}

#endif
