// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -x hip %s
// RUN: not %clang_cc1 -fsyntax-only %s -DSLOC_CHECK 2>&1 | FileCheck %s --strict-whitespace

template<typename T=int> struct S {};
template<typename> void f();

template<typename T, typename... V> struct S<T(V...)> {};

template<typename ...T> struct V {};
template<typename ...T> struct V<void(T)...> {};

void foo(void) {
  // In C++11 mode, all of these are expected to parse correctly, and the CUDA
  // language should not interfere with that.

  // expected-no-diagnostics

  S<S<S<int>>> s3;
  S<S<S<>>> s30;

  S<S<S<S<int>>>> s4;
  S<S<S<S<>>>> s40;

  S<S<S<S<S<int>>>>> s5;
  S<S<S<S<S<>>>>> s50;

  (void)(&f<S<S<int>>>==0);
  (void)(&f<S<S<>>>==0);

  S<S<S<void()>>> s6;
}

template<typename ...T>
void bar(T... args) {
  S<S<V<void(T)...>>> s7;
}

template <typename T, typename T1> void operator<<(T, T1);

struct S1 {};

template <> void operator<<<>(S1, S1);

class C {
public:
  template <typename T> void operator<<(T) {}
};

void foobar() {
  C CC;
  CC.operator<<<int>(1);
  CC.template operator<<<int>(1);
#ifdef SLOC_CHECK
  // We split <<< into a << followed by a <, check that < has right source
  // location.
  CC.operator<<<int;
  // CHECK: [[@LINE-1]]:20: error: expected '>'
  // CHECK-NEXT: CC.operator<<<int;
  // CHECK-NEXT:                  ^
  // CHECK-NEXT: [[@LINE-4]]:16: note: to match this '<'
  // CHECK-NEXT: CC.operator<<<int;
  // CHECK-NEXT:              ^
  CC.template operator<<<int;
  // CHECK: [[@LINE-1]]:29: error: expected '>'
  // CHECK-NEXT: CC.template operator<<<int;
  // CHECK-NEXT:                           ^
  // CHECK-NEXT: [[@LINE-4]]:25: note: to match this '<'
  // CHECK-NEXT: CC.template operator<<<int;
  // CHECK-NEXT:                       ^
#endif
}
