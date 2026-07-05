// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct T { 
  struct x {
    int m;
  };
  x* operator->();
  void operator++(int);
  void operator<<(int);
  T();
  T(int);
  T(int, int);
};

template<typename A, typename B, typename C, typename D, typename E>
void func(A, B, C, D, E);

void func(int a, int c) {
  T(a)->m = 7;
  T(a)++;
  T(a,5)<<c;

  T(*d)(int);
  T(e)[5];
  T(f) = {1, 2};
  T(*g)(double(3)); // expected-error{{cannot initialize a variable of type 'T (*)' with an rvalue of type 'double'}}
  func(a, d, e, f, g);
}

void func2(int a, int c) {
  decltype(T())(a)->m = 7;
  decltype(T())(a)++;
  decltype(T())(a,5)<<c;

  decltype(T())(*d)(int);
  decltype(T())(e)[5];
  decltype(T())(f) = {1, 2};
  decltype(T())(*g)(double(3)); // expected-error{{cannot initialize a variable of type 'decltype(T()) (*)' (aka 'T *') with an rvalue of type 'double'}}
  func(a, d, e, f, g);
}
