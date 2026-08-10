// RUN: not %clang_cc1 -x c++ -std=c++17 -fsyntax-only %s
template <typename U>
class Imp {
  template <typename F>
  explicit Imp(F f);
};

template <typename T>
class Cls {
  explicit Imp() : f() {}
};
