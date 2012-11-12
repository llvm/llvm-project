// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename A> class s0 {

  template<typename B> class s1 : public s0<A> {
    ~s1() {}
    s0<A> ms0;
  };

};

struct Incomplete;

template<typename T>
void destroy_me(T me) {
  me.~T();
}

template void destroy_me(Incomplete*);

namespace PR6152 {
  template<typename T> struct X { void f(); };
  template<typename T> struct Y { };
  template<typename T>
  void X<T>::f() {
    Y<T> *y;
    y->template Y<T>::~Y();
    y->template Y<T>::~Y<T>();
    y->~Y();
  }
  
  template struct X<int>;
}

namespace cvquals {
  template<typename T>
  void f(int *ptr) {
    ptr->~T();
  }

  template void f<const volatile int>(int *);
}

namespace PR7239 {
  template<class E> class A { };
  class B {
    void f() {
      A<int>* x;
      x->A<int>::~A<int>();
    }
  };
}

namespace PR7904 {
  struct Foo {
    template <int i> ~Foo() {} // expected-error{{destructor cannot be declared as a template}}
  };
  Foo f;
}
