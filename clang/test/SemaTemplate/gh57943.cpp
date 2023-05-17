// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics
struct s {
    template<typename T>
          requires requires(T x) { x.g(); }
      friend void f(T);
};
