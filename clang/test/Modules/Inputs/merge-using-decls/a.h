struct X {
  int v;
  typedef int t;
  void f(X);
};

struct YA {
  int value;
  typedef int type;
};

struct Z {
  void f(Z);
};

template<typename T> struct C : X, T {
  using T::value;
  using typename T::type;
  using X::v;
  using typename X::t;
};

template<typename T> struct D : X, T {
  using T::value;
  using typename T::type;
  using X::v;
  using typename X::t;
};

template<typename T> struct E : X, T {
  using T::value;
  using typename T::type;
  using X::v;
  using typename X::t;
};

template<typename T> struct F : X, T {
  using T::value;
  using typename T::type;
  using X::v;
  using typename X::t;
};

// Force instantiation.
typedef C<YA>::type I;
typedef D<YA>::type I;
typedef E<YA>::type I;
typedef F<YA>::type I;

#if __cplusplus >= 201702L
template<typename ...T> struct G : T... {
  using T::f...;
};
using Q = decltype(G<X, Z>());
#endif
