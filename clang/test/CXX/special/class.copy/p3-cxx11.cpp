// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++2b -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++11 -fsyntax-only -verify %s
class X {
  X(const X&);

public:
  X();
  X(X&&);
};

X return_by_move(int i, X x) {
  X x2;
  if (i == 0)
    return x;
  else if (i == 1)
    return x2;
  else
    return x;
}

void throw_move_only(X x) {
  X x2;
  throw x;
  throw x2;
}

namespace PR10142 {
  struct X {
    X();
    X(X&&);
    X(const X&) = delete; // expected-note 2{{'X' has been explicitly marked deleted here}}
  };

  void f(int i) {
    X x;
    try {
      X x2;
      if (i)
        throw x2; // okay
      throw x; // expected-error{{call to deleted constructor of 'PR10142::X'}}
    } catch (...) {
    }
  }

  template<typename T>
  void f2(int i) {
    T x;
    try {
      T x2;
      if (i)
        throw x2; // okay
      throw x; // expected-error{{call to deleted constructor of 'PR10142::X'}}
    } catch (...) {
    }
  }

  template void f2<X>(int); // expected-note{{in instantiation of function template specialization 'PR10142::f2<PR10142::X>' requested here}}
}
