// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx98-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx98-14,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx17,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx17,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx17,since-cxx14,since-cxx11,since-cxx23 -fexceptions -fcxx-exceptions -pedantic-errors

// dr1200: na

namespace dr1213 { // dr1213: 7
#if __cplusplus >= 201103L
  using T = int[3];
  int &&r = T{}[1];

  using T = decltype((T{}));
  using U = decltype((T{}[2]));
  using U = int &&;

  // Same thing but in a case where we consider overloaded operator[].
  struct ConvertsToInt {
    operator int();
  };
  struct X { int array[1]; };
  using U = decltype(X().array[ConvertsToInt()]);

  // We apply the same rule to vector subscripting.
  typedef int V4Int __attribute__((__vector_size__(sizeof(int) * 4)));
  typedef int EV4Int __attribute__((__ext_vector_type__(4)));
  using U = decltype(V4Int()[0]);
  using U = decltype(EV4Int()[0]);
#endif
}

#if __cplusplus >= 201103L
namespace dr1223 { // dr1223: 17 drafting
struct M;
template <typename T>
struct V;
struct S {
  S* operator()();
  int N;
  int M;
#if __cplusplus >= 202302L
  template <typename T>
  static constexpr auto V = 0;
  void f(char);
  void f(int);
  void mem(S s) {
    auto(s)()->M;
    // since-cxx23-warning@-1 {{expression result unused}}
    auto(s)()->V<int>;
    // since-cxx23-warning@-1 {{expression result unused}}
    auto(s)()->f(0);
  }
#endif
};
void f(S s) {
  {
#if __cplusplus >= 202302L
    auto(s)()->N;
    //since-cxx23-warning@-1 {{expression result unused}}
#endif
    auto(s)()->M;
  }
  {
    S(s)()->N;
    // since-cxx11-warning@-1 {{expression result unused}}
    S(s)()->M;
    // since-cxx11-warning@-1 {{expression result unused}}
  }
}

struct A {
    A(int*);
    A *operator()();
};
typedef struct BB { int C[2]; } *B, C;
void g() {
    A a(B ()->C);
    A b(auto ()->C);
    static_assert(sizeof(B ()->C[1] == sizeof(int)), "");
    sizeof(auto () -> C[1]);
    // since-cxx11-error@-1 {{function cannot return array type 'C[1]' (aka 'dr1223::BB[1]')}}
}

}
#endif

#if __cplusplus >= 201103L
namespace dr1227 { // dr1227: 3.0
template <class T> struct A { using X = typename T::X; };
// since-cxx11-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
//   since-cxx11-note@#dr1227-g {{in instantiation of template class 'dr1227::A<int>' requested here}}
//   since-cxx11-note@#dr1227-g-int {{while substituting explicitly-specified template arguments into function template 'g'}}
template <class T> typename T::X f(typename A<T>::X);
template <class T> void f(...) { }
template <class T> auto g(typename A<T>::X) -> typename T::X; // #dr1227-g
template <class T> void g(...) { }

void h() {
  f<int>(0); // OK, substituting return type causes deduction to fail
  g<int>(0); // #dr1227-g-int
}
}
#endif

namespace dr1250 { // dr1250: 3.9
struct Incomplete;

struct Base {
  virtual const Incomplete *meow() = 0;
};

struct Derived : Base {
  virtual Incomplete *meow();
};
}

namespace dr1265 { // dr1265: 5
#if __cplusplus >= 201103L
  auto a = 0, b() -> int;
  // since-cxx11-error@-1 {{declaration with trailing return type must be the only declaration in its group}}
  auto b() -> int, d = 0;
  // since-cxx11-error@-1 {{declaration with trailing return type must be the only declaration in its group}}
  auto e() -> int, f() -> int;
  // since-cxx11-error@-1 {{declaration with trailing return type must be the only declaration in its group}}
#endif

#if __cplusplus >= 201402L
  auto g(), h = 0;
  // since-cxx14-error@-1 {{function with deduced return type must be the only declaration in its group}}
  auto i = 0, j();
  // since-cxx14-error@-1 {{function with deduced return type must be the only declaration in its group}}
  auto k(), l();
  // since-cxx14-error@-1 {{function with deduced return type must be the only declaration in its group}}
#endif
}

// dr1291: na

namespace dr1295 { // dr1295: 4
  struct X {
    unsigned bitfield : 4;
  };

  X x = {1};

  unsigned const &r1 = static_cast<X &&>(x).bitfield;
  // cxx98-error@-1 {{rvalue references are a C++11 extension}}
  unsigned const &r2 = static_cast<unsigned &&>(x.bitfield);
  // cxx98-error@-1 {{rvalue references are a C++11 extension}}

  template<unsigned &r> struct Y {}; // #dr1295-Y
  Y<x.bitfield> y; // #dr1295-y
  // cxx98-14-error@-1 {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#dr1295-Y {{template parameter is declared here}}
  // since-cxx17-error@#dr1295-y {{non-type template argument refers to subobject 'x.bitfield'}}

#if __cplusplus >= 201103L
  const unsigned other = 0;
  using T = decltype(true ? other : x.bitfield);
  using T = unsigned;
#endif
}

