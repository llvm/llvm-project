// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

typedef double A;
template<typename T> class B {
  typedef int A;
};

template<typename T> struct X : B<T> {
  static A a;
};

int a0[sizeof(X<int>::a) == sizeof(double) ? 1 : -1];

// PR4365.
template<class T> class Q;
template<class T> class R : Q<T> {T current;};


namespace test0 {
  template <class T> class Base {
  public:
    void instance_foo();
    static void static_foo();
    class Inner {
    public:
      void instance_foo();
      static void static_foo();
    };
  };

  template <class T> class Derived1 : Base<T> {
  public:
    void test0() {
      Base<T>::static_foo();
      Base<T>::instance_foo();
    }

    void test1() {
      Base<T>::Inner::static_foo();
      Base<T>::Inner::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    static void test2() {
      Base<T>::static_foo();
      Base<T>::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    static void test3() {
      Base<T>::Inner::static_foo();
      Base<T>::Inner::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }
  };

  template <class T> class Derived2 : Base<T>::Inner {
  public:
    void test0() {
      Base<T>::static_foo();
      Base<T>::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    void test1() {
      Base<T>::Inner::static_foo();
      Base<T>::Inner::instance_foo();
    }

    static void test2() {
      Base<T>::static_foo();
      Base<T>::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }

    static void test3() {
      Base<T>::Inner::static_foo();
      Base<T>::Inner::instance_foo(); // expected-error {{call to non-static member function without an object argument}}
    }
  };

  void test0() {
    Derived1<int> d1;
    d1.test0();
    d1.test1(); // expected-note {{in instantiation of member function}}
    d1.test2(); // expected-note {{in instantiation of member function}}
    d1.test3(); // expected-note {{in instantiation of member function}}

    Derived2<int> d2;
    d2.test0(); // expected-note {{in instantiation of member function}}
    d2.test1();
    d2.test2(); // expected-note {{in instantiation of member function}}
    d2.test3(); // expected-note {{in instantiation of member function}}
  }
}

namespace test1 {
  template <class T> struct Base {
    void foo(T); // expected-note {{member is declared here}}
  };

  template <class T> struct Derived : Base<T> {
    void doFoo(T v) {
      foo(v); // expected-error {{explicit qualification required to use member 'foo' from dependent base class}}
    }
  };

  template struct Derived<int>; // expected-note {{requested here}}
}

namespace PR8966 {
  template <class T>
  class MyClassCore
  {
  };

  template <class T>
  class MyClass : public MyClassCore<T>
  {
  public:
    enum  {
      N
    };

    // static member declaration
    static const char* array [N];

    void f() {
      MyClass<T>::InBase = 17;
    }
  };

  // static member definition
  template <class T>
  const char* MyClass<T>::array [MyClass<T>::N] = { "A", "B", "C" };
}

namespace std {
  inline namespace v1 {
    template<typename T> struct basic_ostream;
  }
  namespace inner {
    template<typename T> struct vector {};
  }
  using inner::vector;
  template<typename T, typename U> struct pair {};
  typedef basic_ostream<char> ostream;
  extern ostream cout;
  std::ostream &operator<<(std::ostream &out, const char *);
}

namespace PR10053 {
  template<typename T> struct A {
    T t;
    A() {
      f(t); // expected-error {{call to function 'f' that is neither visible in the template definition nor found by argument-dependent lookup}}
    }
  };

  void f(int&); // expected-note {{'f' should be declared prior to the call site}}

  A<int> a; // expected-note {{in instantiation of member function}}


  namespace N {
    namespace M {
      template<typename T> int g(T t) {
        f(t); // expected-error {{call to function 'f' that is neither visible in the template definition nor found by argument-dependent lookup}}
      };
    }

    void f(char&); // expected-note {{'f' should be declared prior to the call site}}
  }

  void f(char&);

  int k = N::M::g<char>(0);; // expected-note {{in instantiation of function}}


  namespace O {
    int f(char&); // expected-note {{candidate function not viable}}

    template<typename T> struct C {
      static const int n = f(T()); // expected-error {{no matching function}}
    };
  }

  int f(double); // no note, shadowed by O::f
  O::C<double> c; // expected-note {{requested here}}


  // Example from www/compatibility.html
  namespace my_file {
    template <typename T> T Squared(T x) {
      return Multiply(x, x); // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
    }

    int Multiply(int x, int y) { // expected-note {{should be declared prior to the call site}}
      return x * y;
    }

    int main() {
      Squared(5); // expected-note {{here}}
    }
  }

  // Example from www/compatibility.html
  namespace my_file2 {
    template<typename T>
    void Dump(const T& value) {
      std::cout << value << "\n"; // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
    }

    namespace ns {
      struct Data {};
    }

    std::ostream& operator<<(std::ostream& out, ns::Data data) { // expected-note {{should be declared prior to the call site or in namespace 'PR10053::my_file2::ns'}}
      return out << "Some data";
    }

    void Use() {
      Dump(ns::Data()); // expected-note {{here}}
    }
  }

  namespace my_file2_a {
    template<typename T>
    void Dump(const T &value) {
      print(std::cout, value); // expected-error 4{{neither visible in the template definition nor found by argument-dependent lookup}}
    }

    namespace ns {
      struct Data {};
    }
    namespace ns2 {
      struct Data {};
    }

    std::ostream &print(std::ostream &out, int); // expected-note-re {{should be declared prior to the call site{{$}}}}
    std::ostream &print(std::ostream &out, ns::Data); // expected-note {{should be declared prior to the call site or in namespace 'PR10053::my_file2_a::ns'}}
    std::ostream &print(std::ostream &out, std::vector<ns2::Data>); // expected-note {{should be declared prior to the call site or in namespace 'PR10053::my_file2_a::ns2'}}
    std::ostream &print(std::ostream &out, std::pair<ns::Data, ns2::Data>); // expected-note {{should be declared prior to the call site or in an associated namespace of one of its arguments}}

    void Use() {
      Dump(0); // expected-note {{requested here}}
      Dump(ns::Data()); // expected-note {{requested here}}
      Dump(std::vector<ns2::Data>()); // expected-note {{requested here}}
      Dump(std::pair<ns::Data, ns2::Data>()); // expected-note {{requested here}}
    }
  }

  namespace unary {
    template<typename T>
    T Negate(const T& value) {
      return !value; // expected-error {{call to function 'operator!' that is neither visible in the template definition nor found by argument-dependent lookup}}
    }

    namespace ns {
      struct Data {};
    }

    ns::Data operator!(ns::Data); // expected-note {{should be declared prior to the call site or in namespace 'PR10053::unary::ns'}}

    void Use() {
      Negate(ns::Data()); // expected-note {{requested here}}
    }
  }
}

namespace PR10187 {
  namespace A1 {
    template<typename T>
    struct S {
      void f() {
        for (auto &a : e)
          __range(a); // expected-error {{undeclared identifier '__range'}}
      }
      int e[10];
    };
  }

  namespace A2 {
    template<typename T>
    struct S {
      void f() {
        for (auto &a : e)
          __range(a); // expected-error {{undeclared identifier '__range'}}
      }
      T e[10];
    };
    void g() {
      S<int>().f(); // expected-note {{here}}
    }
    struct X {};
    void __range(X);
    void h() {
      S<X>().f();
    }
  }

  namespace B {
    template<typename T> void g(); // expected-note {{not viable}}
    template<typename T> void f() {
      g<int>(T()); // expected-error {{no matching function}}
    }

    namespace {
      struct S {};
    }
    void g(S);

    template void f<S>(); // expected-note {{here}}
  }
}

namespace rdar11242625 {

template <typename T>
struct Main {
  struct default_names {
    typedef int id;
  };

  template <typename T2 = typename default_names::id>
  struct TS {
    T2 q;
  };
};

struct Sub : public Main<int> {
  TS<> ff;
};

int arr[sizeof(Sub)];

}

namespace PR11421 {
template < unsigned > struct X {
  static const unsigned dimension = 3;
  template<unsigned dim=dimension>
  struct Y: Y<dim> { }; // expected-error{{base class has incomplete type}}
                        // expected-note@-1{{definition of 'Y<dim>' is not complete until the closing '}'}}
};
typedef X<3> X3;
X3::Y<>::iterator it; // expected-error {{no type named 'iterator' in 'PR11421::X<3>::Y<>'}}
}

namespace rdar12629723 {
  template<class T>
  struct X {
    struct C : public C { }; // expected-error{{base class has incomplete type}}
                             // expected-note@-1{{definition of 'rdar12629723::X::C' is not complete until the closing '}'}}

    struct B;

    struct A : public B {
      virtual void foo() { }
    };

    struct D : T::foo { };
    struct E : D { };
  };

  template<class T>
  struct X<T>::B : public A {
    virtual void foo() { }
  };
}

namespace test_reserved_identifiers {
  template<typename A, typename B> void tempf(A a, B b) {
    a + b;  // expected-error{{call to function 'operator+' that is neither visible in the template definition nor found by argument-dependent lookup}}
  }
  namespace __gnu_cxx { struct X {}; }
  namespace ns { struct Y {}; }
  void operator+(__gnu_cxx::X, ns::Y);  // expected-note{{or in namespace 'test_reserved_identifiers::ns'}}
  void test() {
    __gnu_cxx::X x;
    ns::Y y;
    tempf(x, y);  // expected-note{{in instantiation of}}
  }
}

// This test must live in the global namespace.
struct PR14695_X {};
// FIXME: This note is bogus; it is the using directive which would need to move
// to prior to the call site to fix the problem.
namespace PR14695_A { void PR14695_f(PR14695_X); } // expected-note {{'PR14695_f' should be declared prior to the call site or in the global namespace}}
template<typename T> void PR14695_g(T t) { PR14695_f(t); } // expected-error {{call to function 'PR14695_f' that is neither visible in the template definition nor found by argument-dependent lookup}}
using namespace PR14695_A;
template void PR14695_g(PR14695_X); // expected-note{{requested here}}

namespace OperatorNew {
  template<typename T> void f(T t) {
    operator new(100, t); // expected-error{{call to function 'operator new' that is neither visible in the template definition nor found by argument-dependent lookup}}
    // FIXME: This should give the same error.
    new (t) int;
  }
  struct X {};
};
using size_t = decltype(sizeof(0));
void *operator new(size_t, OperatorNew::X); // expected-note-re {{should be declared prior to the call site{{$}}}}
template void OperatorNew::f(OperatorNew::X); // expected-note {{instantiation of}}

namespace PR19936 {
  template<typename T> decltype(*T()) f() {} // expected-note {{previous}}
  template<typename T> decltype(T() * T()) g() {} // expected-note {{previous}}

  // Create some overloaded operators so we build an overload operator call
  // instead of a builtin operator call for the dependent expression.
  enum E {};
  int operator*(E);
  int operator*(E, E);

  // Check that they still profile the same.
  template<typename T> decltype(*T()) f() {} // expected-error {{redefinition}}
  template<typename T> decltype(T() * T()) g() {} // expected-error {{redefinition}}
}

template <typename> struct CT2 {
  template <class U> struct X;
};
template <typename T> int CT2<int>::X<>; // expected-error {{template parameter list matching the non-templated nested type 'CT2<int>' should be empty}}

namespace DependentTemplateIdWithNoArgs {
  template<typename T> void f() { T::template f(); }
  struct X {
    template<int = 0> static void f();
  };
  void g() { f<X>(); }
}

namespace DependentUnresolvedUsingTemplate {
  template<typename T>
  struct X : T {
    using T::foo;
    void f() { this->template foo(); } // expected-error {{does not refer to a template}}
    void g() { this->template foo<>(); } // expected-error {{does not refer to a template}}
    void h() { this->template foo<int>(); } // expected-error {{does not refer to a template}}
  };
  struct A { template<typename = int> int foo(); };
  struct B { int foo(); }; // expected-note 3{{non-template here}}
  void test(X<A> xa, X<B> xb) {
    xa.f();
    xa.g();
    xa.h();
    xb.f(); // expected-note {{instantiation of}}
    xb.g(); // expected-note {{instantiation of}}
    xb.h(); // expected-note {{instantiation of}}
  }
}

namespace PR37680 {
  template <class a> struct b : a {
    using a::add;
    template<int> int add() { return this->template add(0); }
  };
  struct a {
    template<typename T = void> int add(...);
    void add(int);
  };
  int f(b<a> ba) { return ba.add<0>(); }
}
