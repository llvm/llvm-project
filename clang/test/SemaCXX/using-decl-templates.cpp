// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T> struct A {
  void f() { }
  struct N { }; // expected-note{{target of using declaration}}
};

template<typename T> struct B : A<T> {
  using A<T>::f;
  using A<T>::N; // expected-error{{dependent using declaration resolved to type without 'typename'}}
  
  using A<T>::foo; // expected-error{{no member named 'foo'}}
  using A<double>::f; // expected-error{{using declaration refers into 'A<double>::', which is not a base class of 'B<int>'}}
};

B<int> a; // expected-note{{in instantiation of template class 'B<int>' requested here}}

template<typename T> struct C : A<T> {
  using A<T>::f;
  
  void f() { };
};

template <typename T> struct D : A<T> {
  using A<T>::f;
  
  void f();
};

template<typename T> void D<T>::f() { }

template<typename T> struct E : A<T> {
  using A<T>::f;

  void g() { f(); }
};

namespace test0 {
  struct Base {
    int foo;
  };
  template<typename T> struct E : Base {
    using Base::foo;
  };

  template struct E<int>;
}

// PR7896
namespace PR7896 {
template <class T> struct Foo {
  int k (float);
};
struct Baz {
  int k (int);
};
template <class T> struct Bar : public Foo<T>, Baz {
  using Foo<T>::k;
  using Baz::k;
  int foo() {
    return k (1.0f);
  }
};
template int Bar<int>::foo();
}

// PR10883
namespace PR10883 {
  template <typename T>
  class Base {
   public:
    typedef long Container;
  };

  template <typename T>
  class Derived : public Base<T> {
   public:
    using Base<T>::Container;

    void foo(const Container& current); // expected-error {{unknown type name 'Container'}}
  };
}

template<typename T> class UsingTypenameNNS {
  using typename T::X;
  typename X::X x;
};

namespace aliastemplateinst {
  template<typename T> struct A { };
  template<typename T> using APtr = A<T*>; // expected-note{{previous use is here}}

  template struct APtr<int>; // expected-error{{elaborated type refers to a non-tag type}}
}
