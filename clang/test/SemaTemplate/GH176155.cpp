// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template <int> struct bad {
  template <class T, auto =
                         [] {
                           for (int i = 0; i < 100; ++i) {
                             struct LoopHelper {
                               static constexpr void process() {}
                             };
                           }
                         }>
  static void f(T) {}
};

int main() { bad<0>::f(0); }
