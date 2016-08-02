// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

class C {
    struct S; // expected-note {{previously declared 'private' here}}
public:
    
    struct S {}; // expected-error {{'S' redeclared with 'public' access}}
};

struct S {
    class C; // expected-note {{previously declared 'public' here}}
    
private:
    class C { }; // expected-error {{'C' redeclared with 'private' access}}
};

class T {
protected:
    template<typename T> struct A; // expected-note {{previously declared 'protected' here}}
    
private:
    template<typename T> struct A {}; // expected-error {{'A' redeclared with 'private' access}}
};

// PR5573
namespace test1 {
  class A {
  private:
    class X; // expected-note {{previously declared 'private' here}} \
             // expected-note {{previous declaration is here}}
  public:
    class X; // expected-error {{'X' redeclared with 'public' access}} \
             // expected-warning {{class member cannot be redeclared}}
    class X {};
  };
}

// PR15209
namespace PR15209 {
  namespace alias_templates {
    template<typename T1, typename T2> struct U { };
    template<typename T1> using W = U<T1, float>;

    class A {
      typedef int I;
      static constexpr I x = 0; // expected-note {{implicitly declared private here}}
      static constexpr I y = 42; // expected-note {{implicitly declared private here}}
      friend W<int>;
    };

    template<typename T1>
    struct U<T1, float>  {
      int v_;
      // the following will trigger for U<float, float> instantiation, via W<float>
      U() : v_(A::x) { } // expected-error {{'x' is a private member of 'PR15209::alias_templates::A'}}
    };

    template<typename T1>
    struct U<T1, int> {
      int v_;
      U() : v_(A::y) { } // expected-error {{'y' is a private member of 'PR15209::alias_templates::A'}}
    };

    template struct U<int, int>; // expected-note {{in instantiation of member function 'PR15209::alias_templates::U<int, int>::U' requested here}}

    void f()
    {
      W<int>();
      // we should issue diagnostics for the following
      W<float>(); // expected-note {{in instantiation of member function 'PR15209::alias_templates::U<float, float>::U' requested here}}
    }
  }

  namespace templates {
    class A {
      typedef int I;  // expected-note {{implicitly declared private here}}
      static constexpr I x = 0; // expected-note {{implicitly declared private here}}

      template<int> friend struct B;
      template<int> struct C;
      template<template<int> class T> friend struct TT;
      template<typename T> friend void funct(T);
    };
    template<A::I> struct B { };

    template<A::I> struct A::C { };

    template<template<A::I> class T> struct TT {
      T<A::x> t;
    };

    template struct TT<B>;
    template<A::I> struct D { };  // expected-error {{'I' is a private member of 'PR15209::templates::A'}}
    template struct TT<D>;

    // function template case
    template<typename T>
    void funct(T)
    {
      (void)A::x;
    }

    template void funct<int>(int);

    void f()
    {
      (void)A::x;  // expected-error {{'x' is a private member of 'PR15209::templates::A'}}
    }
  }
}

namespace PR7434 {
  namespace comment0 {
    template <typename T> struct X;
    namespace N {
    class Y {
      template<typename T> friend struct X;
      int t; // expected-note {{here}}
    };
    }
    template<typename T> struct X {
      X() { (void)N::Y().t; } // expected-error {{private}}
    };
    X<char> x;
  }
  namespace comment2 {
    struct X;
    namespace N {
    class Y {
      friend struct X;
      int t; // expected-note {{here}}
    };
    }
    struct X {
      X() { (void)N::Y().t; } // expected-error {{private}}
    };
  }
}

namespace LocalExternVar {
  class test {
  private:
    struct private_struct { // expected-note 2{{here}}
      int x;
    };
    int use_private();
  };

  int test::use_private() {
    extern int array[sizeof(test::private_struct)]; // ok
    return array[0];
  }

  int f() {
    extern int array[sizeof(test::private_struct)]; // expected-error {{private}}
    return array[0];
  }

  int array[sizeof(test::private_struct)]; // expected-error {{private}}
}
