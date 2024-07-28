// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -fcoverage-mapping %s
// expected-no-diagnostics

// GH62105 demonstrated a crash with this example code when calculating
// coverage mapping because some source location information was being dropped.
// Demonstrate that we do not crash on this code.
namespace std { template <typename E> class initializer_list { const E *a, *b; }; }

template <typename> struct T {
  T(std::initializer_list<int>, int = int());
  bool b;
};

template <typename> struct S1 {
  static void foo() {
    class C;
    (void)(0 ? T<C>{} : T<C>{});
  }
};

void bar() {
  S1<int>::foo();
}

