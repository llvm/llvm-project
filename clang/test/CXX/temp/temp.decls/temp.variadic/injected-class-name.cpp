// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check for declaration matching with out-of-line declarations and
// variadic templates, which involves proper computation of the
// injected-class-name.
template<typename T, typename ...Types>
struct X0 {
  typedef T type;

  void f0(T);
  type f1(T);
};

template<typename T, typename ...Types>
void X0<T, Types...>::f0(T) { }

template<typename T, typename ...Types>
typename X0<T, Types...>::type X0<T, Types...>::f1(T) { }

template<typename T, typename ...Types>
struct X0<T, T, Types...> {
  typedef T* result;
  result f3();

  template<typename... InnerTypes>
  struct Inner;
};

template<typename T, typename ...Types>
typename X0<T, T, Types...>::result X0<T, T, Types...>::f3() { return 0; }

template<typename T, typename ...Types>
template<typename ...InnerTypes>
struct X0<T, T, Types...>::Inner {
  template<typename ...ReallyInner> void f4();
};

template<typename T, typename ...Types>
template<typename ...InnerTypes>
template<typename ...ReallyInner>
void X0<T, T, Types...>::Inner<InnerTypes...>::f4() { }

namespace rdar8848837 {
  // Out-of-line definitions that cause rebuilding in the current
  // instantiation.
  template<typename F> struct X;

  template<typename R, typename ...ArgTypes>
  struct X<R(ArgTypes...)> {
    X<R(ArgTypes...)> f();
  };

  template<typename R, typename ...ArgTypes>
  X<R(ArgTypes...)> X<R(ArgTypes...)>::f() { return *this; }


  X<int(float, double)> xif;

  template<unsigned> struct unsigned_c { };
  template<typename ...ArgTypes> int g(ArgTypes...);

  template<typename F> struct X1;

  template<typename R, typename ...ArgTypes>
  struct X1<R(ArgTypes...)> {
    unsigned_c<sizeof(1 + g(ArgTypes()...))> f();
  };

  template<typename R, typename ...ArgTypes>
  unsigned_c<sizeof(1 + g(ArgTypes()...))> X1<R(ArgTypes...)>::f() { 
    return unsigned_c<sizeof(int)>();
  }

  X1<int(float, double)> xif2;
}
