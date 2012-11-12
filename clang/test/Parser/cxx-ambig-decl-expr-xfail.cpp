// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *
struct X { 
  template<typename T> X(T);
  X(int, int);

  X operator()(int, int) const;
};

template<typename T, typename U> struct Y { };

X *x;
void f() {
  int y = 0;
  X (*x)(int(y), int(y)) = Y<int, float>(), ++y;
}
