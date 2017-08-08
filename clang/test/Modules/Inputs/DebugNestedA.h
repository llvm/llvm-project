/* -*- C++ -*- */
template <typename T> class Base {};
template <typename T> struct A : public Base<A<T>> {
  void f();
};

class F {};
typedef A<F> AF;
