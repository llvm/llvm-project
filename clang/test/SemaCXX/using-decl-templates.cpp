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

  template struct APtr<int>; // expected-error{{alias template 'APtr' cannot be referenced with the 'struct' specifier}}
}

namespace DontDiagnoseInvalidTest {
template <bool Value> struct Base {
  static_assert(Value, ""); // expected-error {{static assertion failed}}
};
struct Derived : Base<false> { // expected-note {{requested here}}
  using Base<false>::Base; // OK. Don't diagnose that 'Base' isn't a base class of Derived.
};
} // namespace DontDiagnoseInvalidTest

namespace shadow_nested_operator {
template <typename T>
struct A {
  struct Nested {};
  operator Nested*() {return 0;};
};

template <typename T>
struct B : A<T> {
  using A<T>::operator typename A<T>::Nested*;
  operator typename A<T>::Nested *() {
    struct A<T> * thi = this;
    return *thi;
 };
};

int foo () {
  struct B<int> b;
  auto s = *b;
}
} // namespace shadow_nested_operator

namespace func_templ {
namespace sss {
double foo(int, double);
template <class T>
T foo(T);
} // namespace sss

namespace oad {
void foo();
}

namespace oad {
using sss::foo;
}

namespace sss {
using oad::foo;
}

namespace sss {
double foo(int, double) { return 0; }
// There used to be an error with the below declaration when the example should
// be accepted.
template <class T>
T foo(T t) { // OK
  return t;
}
} // namespace sss
} // namespace func_templ
