// RUN: %clang_cc1 -fsyntax-only -verify %s

// This test concerns the identity of dependent types within the
// canonical type system, specifically focusing on the difference
// between members of the current instantiation and members of an
// unknown specialization. This considers C++ [temp.type], which
// specifies type equivalence within a template, and C++0x
// [temp.dep.type], which defines what it means to be a member of the
// current instantiation.

template<typename T, typename U>
struct X0 {
  typedef T T_type;
  typedef U U_type;

  void f0(T&); // expected-note{{previous}}
  void f0(typename X0::U_type&);
  void f0(typename X0::T_type&); // expected-error{{redecl}}

  void f1(T&); // expected-note{{previous}}
  void f1(typename X0::U_type&);
  void f1(typename X0<T, U>::T_type&); // expected-error{{redecl}}

  void f2(T&); // expected-note{{previous}}
  void f2(typename X0::U_type&);
  void f2(typename X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

  void f3(T&); // expected-note{{previous}}
  void f3(typename X0::U_type&);
  void f3(typename ::X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

  struct X1 {
    typedef T my_T_type;

    void g0(T&); // expected-note{{previous}}
    void g0(typename X0::U_type&);
    void g0(typename X0::T_type&); // expected-error{{redecl}}

    void g1(T&); // expected-note{{previous}}
    void g1(typename X0::U_type&);
    void g1(typename X0<T, U>::T_type&); // expected-error{{redecl}}

    void g2(T&); // expected-note{{previous}}
    void g2(typename X0::U_type&);
    void g2(typename X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

    void g3(T&); // expected-note{{previous}}
    void g3(typename X0::U_type&);
    void g3(typename ::X0<T_type, U_type>::T_type&); // expected-error{{redecl}}

    void g4(T&); // expected-note{{previous}}
    void g4(typename X0::U_type&);
    void g4(typename X1::my_T_type&); // expected-error{{redecl}}

    void g5(T&); // expected-note{{previous}}
    void g5(typename X0::U_type&);
    void g5(typename X0::X1::my_T_type&); // expected-error{{redecl}}

    void g6(T&); // expected-note{{previous}}
    void g6(typename X0::U_type&);
    void g6(typename X0<T, U>::X1::my_T_type&); // expected-error{{redecl}}

    void g7(T&); // expected-note{{previous}}
    void g7(typename X0::U_type&);
    void g7(typename ::X0<typename X1::my_T_type, U_type>::X1::my_T_type&); // expected-error{{redecl}}

    void g8(T&); // expected-note{{previous}}
    void g8(typename X0<U, T_type>::T_type&);
    void g8(typename ::X0<typename X0<T_type, U>::X1::my_T_type, U_type>::X1::my_T_type&); // expected-error{{redecl}}
  };
};


template<typename T, typename U>
struct X0<T*, U*> {
  typedef T T_type;
  typedef U U_type;
  typedef T* Tptr;
  typedef U* Uptr;

  void f0(T&); // expected-note{{previous}}
  void f0(typename X0::U_type&);
  void f0(typename X0::T_type&); // expected-error{{redecl}}

  void f1(T&); // expected-note{{previous}}
  void f1(typename X0::U_type&);
  void f1(typename X0<T*, U*>::T_type&); // expected-error{{redecl}}

  void f2(T&); // expected-note{{previous}}
  void f2(typename X0::U_type&);
  void f2(typename X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}

  void f3(T&); // expected-note{{previous}}
  void f3(typename X0::U_type&);
  void f3(typename ::X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}

  void f4(T&); // expected-note{{previous}}
  void f4(typename X0::U_type&);
  void f4(typename ::X0<Tptr, Uptr>::T_type&); // expected-error{{redecl}}

  void f5(X0*); // expected-note{{previous}}
  void f5(::X0<T, U>*);
  void f5(::X0<T*, U*>*); // expected-error{{redecl}}

  struct X2 {
    typedef T my_T_type;

    void g0(T&); // expected-note{{previous}}
    void g0(typename X0::U_type&);
    void g0(typename X0::T_type&); // expected-error{{redecl}}

    void g1(T&); // expected-note{{previous}}
    void g1(typename X0::U_type&);
    void g1(typename X0<T*, U*>::T_type&); // expected-error{{redecl}}

    void g2(T&); // expected-note{{previous}}
    void g2(typename X0::U_type&);
    void g2(typename X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}

    void g3(T&); // expected-note{{previous}}
    void g3(typename X0::U_type&);
    void g3(typename ::X0<T_type*, U_type*>::T_type&); // expected-error{{redecl}}

    void g4(T&); // expected-note{{previous}}
    void g4(typename X0::U_type&);
    void g4(typename X2::my_T_type&); // expected-error{{redecl}}

    void g5(T&); // expected-note{{previous}}
    void g5(typename X0::U_type&);
    void g5(typename X0::X2::my_T_type&); // expected-error{{redecl}}

    void g6(T&); // expected-note{{previous}}
    void g6(typename X0::U_type&);
    void g6(typename X0<T*, U*>::X2::my_T_type&); // expected-error{{redecl}}

    void g7(T&); // expected-note{{previous}}
    void g7(typename X0::U_type&);
    void g7(typename ::X0<typename X2::my_T_type*, U_type*>::X2::my_T_type&); // expected-error{{redecl}}

    void g8(T&); // expected-note{{previous}}
    void g8(typename X0<U, T_type>::T_type&);
    void g8(typename ::X0<typename X0<T_type*, U*>::X2::my_T_type*, U_type*>::X2::my_T_type&); // expected-error{{redecl}}
  };
};

template<typename T>
struct X1 {
  static int *a;
  void f(float *b) {
    X1<T>::a = b; // expected-error{{incompatible}}
    X1<T*>::a = b;
  }
};

namespace ConstantInCurrentInstantiation {
  template<typename T>
  struct X {
    static const int value = 2;
    static int array[value];
  };

  template<typename T> const int X<T>::value;

  template<typename T>
  int X<T>::array[X<T>::value] = { 1, 2 };
}

namespace Expressions {
  template <bool b>
  struct Bool {
    enum anonymous_enum { value = b };
  };
  struct True : public Bool<true> {};
  struct False : public Bool<false> {};

  template <typename T1, typename T2>
  struct Is_Same : public False {};
  template <typename T>
  struct Is_Same<T, T> : public True {};

  template <bool b, typename T = void>
  struct Enable_If {};
  template <typename T>
  struct Enable_If<true, T>  {
    typedef T type;
  };

  template <typename T>
  class Class {
  public:
    template <typename U>
    typename Enable_If<Is_Same<U, Class>::value, void>::type
    foo();
  };


  template <typename T>
  template <typename U>
  typename Enable_If<Is_Same<U, Class<T> >::value, void>::type
  Class<T>::foo() {}
}

namespace PR9255 {
  template<typename T>
  class X0  {
  public:
    class Inner1;

    class Inner2  {
    public:
      void f()
      {
        Inner1::f.g();
      }
    };
  };
}

namespace rdar10194295 {
  template<typename XT>
  class X {
  public:
    enum Enum { Yes, No };
    template<Enum> void foo();
    template<Enum> class Inner;
  };

  template<typename XT>
  template<typename X<XT>::Enum>
  void X<XT>::foo()
  {
  }

  template<typename XT>
  template<typename X<XT>::Enum>
  class X<XT>::Inner { };
}

namespace RebuildDependentScopeDeclRefExpr {
  template<int> struct N {};
  template<typename T> struct X {
    static const int thing = 0;
    N<thing> data();
    N<thing> foo();
  };
  template<typename T> N<X<T>::thing> X<T>::data() {}
  // FIXME: We should issue a typo-correction here.
  template<typename T> N<X<T>::think> X<T>::foo() {} // expected-error {{no member named 'think' in 'RebuildDependentScopeDeclRefExpr::X<T>'}}
}
