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
    void foo(T); // expected-note {{must qualify identifier to find this declaration in dependent base class}}
  };

  template <class T> struct Derived : Base<T> {
    void doFoo(T v) {
      foo(v); // expected-error {{use of undeclared identifier}}
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
    void f(char&); // expected-note {{candidate function not viable}}

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

    std::ostream &print(std::ostream &out, int); // expected-note-re {{should be declared prior to the call site$}}
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
  namespace A {
    template<typename T>
    struct S {
      void f() {
        for (auto &a : e)
          __range(a); // expected-error {{undeclared identifier '__range'}}
      }
      int e[10];
    };
    void g() {
      S<int>().f(); // expected-note {{here}}
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
  struct Y: Y<dim> { }; // expected-error {{incomplete type}} expected-note {{is not complete until the closing}}
};
typedef X<3> X3;
X3::Y<>::iterator it; // expected-note {{requested here}}
}
