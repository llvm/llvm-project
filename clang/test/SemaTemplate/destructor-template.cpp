// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename A> class s0 {
  template<typename B> class s1;
};

template<typename A>
template<typename B>
class s0<A>::s1 : s0<A> {
  ~s1() {}
  s0<A> ms0;
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
  struct Foo {};
  template <class T>
  Foo::~Foo() { // expected-error{{destructor cannot be declared as a template}}
    T t;
    T &pT = t;
    pT;
  }
  Foo f;
}

namespace rdar13140795 {
  template <class T> class shared_ptr {};

  template <typename T> struct Marshal {
    static int gc();
  };


  template <typename T> int Marshal<T>::gc() {
    shared_ptr<T> *x;
    x->template shared_ptr<T>::~shared_ptr();
    return 0;
  }

  void test() {
    Marshal<int>::gc();
  }
}

namespace PR16852 {
  template<typename T> struct S { int a; T x; };
  template<typename T> decltype(S<T>().~S()) f(); // expected-note {{candidate template ignored: couldn't infer template argument 'T'}}
  void g() { f(); } // expected-error {{no matching function for call to 'f'}}
}

class PR33189
{
  template <class T>
  ~PR33189() { } // expected-error{{destructor cannot be declared as a template}}
};

namespace PR38671 {
struct S {
  template <class>
  ~S(); // expected-error{{destructor cannot be declared as a template}}
};
struct T : S {
  ~T() = default;
};
} // namespace PR38671
