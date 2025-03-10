// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
template<typename T, int N = 2> struct X; // expected-note {{template parameter is declared here}}

X<int, 1> *x1;
X<int> *x2;

X<> *x3; // expected-error{{missing template argument for template parameter}}

template<typename U = float, int M> struct X;

X<> *x4;

template<typename T = int> struct Z { };
template struct Z<>;

// PR4362
template<class T> struct a { };
template<> struct a<int> { static const bool v = true; };

template<class T, bool = a<T>::v> struct p { }; // expected-error {{no member named 'v'}}
                                                // expected-note@-1 {{template parameter is declared here}}

template struct p<bool>; // expected-note {{in instantiation of default argument for 'p<bool>' required here}}
template struct p<int>;

// PR5187
template<typename T, typename U>
struct A;

template<typename T, typename U = T>
struct A;

template<typename T, typename U>
struct A {
  void f(A<T>);
};

template<typename T>
struct B { };

template<>
struct B<void> {
  typedef B<void*> type;
};

// Nested default arguments for template parameters.
template<typename T> struct X1 { };

template<typename T>
struct X2 {
  template<typename U = typename X1<T>::type> // expected-error{{no type named 'type' in 'X1<int>'}} \
                                              // expected-error{{no type named 'type' in 'X1<char>'}} \
                                              // expected-note {{template parameter is declared here}}
  struct Inner1 { };

  template<T Value = X1<T>::value> // expected-error{{no member named 'value' in 'X1<int>'}} \
                                   // expected-error{{no member named 'value' in 'X1<char>'}} \
                                   // expected-note {{template parameter is declared here}}
  struct NonType1 { };

  template<T Value>
  struct Inner2 { };

  template<typename U>
  struct Inner3 {
    template<typename X = T, typename V = U>
    struct VeryInner { };

    template<T Value1 = sizeof(T), T Value2 = sizeof(U),
             T Value3 = Value1 + Value2>
    struct NonType2 { };
  };
};

X2<int> x2i; // expected-note{{in instantiation of template class 'X2<int>' requested here}}
X2<int>::Inner1<float> x2iif;

X2<int>::Inner1<> x2bad; // expected-error{{missing template argument for template parameter}}

X2<int>::NonType1<'a'> x2_nontype1;
X2<int>::NonType1<> x2_nontype1_bad; // expected-error{{missing template argument for template parameter}}

// Check multi-level substitution into template type arguments
X2<int>::Inner3<float>::VeryInner<> vi;
X2<char>::Inner3<int>::NonType2<> x2_deep_nontype; // expected-note{{in instantiation of template class 'X2<char>' requested here}}

template<typename T, typename U>
struct is_same { static const bool value = false; };

template<typename T>
struct is_same<T, T> { static const bool value = true; };

int array1[is_same<__typeof__(vi),
               X2<int>::Inner3<float>::VeryInner<int, float> >::value? 1 : -1];

int array2[is_same<__typeof(x2_deep_nontype),
                   X2<char>::Inner3<int>::NonType2<sizeof(char), sizeof(int),
                                    sizeof(char)+sizeof(int)> >::value? 1 : -1];

// Template template parameter defaults
template<template<typename T> class X = X2> struct X3 { };
int array3[is_same<X3<>, X3<X2> >::value? 1 : -1];

struct add_pointer {
  template<typename T>
  struct apply {
    typedef T* type;
  };
};

template<typename T, template<typename> class X = T::template apply>
  struct X4;
int array4[is_same<X4<add_pointer>,
                   X4<add_pointer, add_pointer::apply> >::value? 1 : -1];

template<int> struct X5 {}; // expected-note {{template parameter is declared here}}
template<long long> struct X5b {};
template<typename T,
         template<T> class B = X5> // expected-error {{cannot be narrowed from type 'long long' to 'int'}}
                                   // expected-note@-1 {{template template argument is incompatible}}
                                   // expected-note@-2 {{template parameter is declared here}}
  struct X6 {};

X6<int> x6a;
X6<long long> x6b; // expected-note {{while checking a default template argument used here}}
X6<long long, X5b> x6c;


template<template<class> class X = B<int> > struct X7; // expected-error{{must be a class template}}

namespace PR9643 {
  template<typename T> class allocator {};
  template<typename T, typename U = allocator<T> > class vector {};

  template<template<typename U, typename = allocator<U> > class container,
           typename DT>
  container<DT> initializer(const DT& d) {
    return container<DT>();
  }

  void f() {
    vector<int, allocator<int> > v = initializer<vector>(5);
  }
}

namespace PR16288 {
  template<typename X>
  struct S {
    template<typename T = int, typename U>
#if __cplusplus <= 199711L // C++03 or earlier modes
    // expected-warning@-2 {{default template arguments for a function template are a C++11 extension}}
#endif
    void f();
  };
  template<typename X>
  template<typename T, typename U>
  void S<X>::f() {}
}

namespace DR1635 {
  template <class T> struct X {
    template <class U = typename T::type> static void f(int) {} // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
#if __cplusplus <= 199711L // C++03 or earlier modes
    // expected-warning@-2 {{default template arguments for a function template are a C++11 extension}}
#endif
    static void f(...) {}
  };

  int g() { X<int>::f(0); } // expected-note {{in instantiation of template class 'DR1635::X<int>' requested here}}
}

namespace NondefDecls {
  template<typename T> void f1() {
    int g1(int defarg = T::error);  // expected-error{{type 'int' cannot be used prior to '::' because it has no members}} \
                                    // expected-note {{in instantiation of default function argument expression for 'g1<int>' required here}}
  }
  template void f1<int>();  // expected-note{{in instantiation of function template specialization 'NondefDecls::f1<int>' requested here}}
}

template <typename T>
struct C {
  C(T t = ); // expected-error {{expected expression}}
};
C<int> obj;

namespace PR26134 {
// Make sure when substituting default template arguments we do it in the current context.
template<class T, bool Val = T::value>
struct X {};

template<bool B> struct Y {
  void f() { X<Y> xy; }
  static const bool value = B;
};

namespace ns1 {
template<class T0>
struct X {
  template<bool B = T0::value> struct XInner { static const bool value = B; };
};
template<bool B> struct S { static const bool value = B; };
#if __cplusplus > 199711L
template<bool B> struct Y {
  static constexpr bool f() { return typename X<S<B>>::template XInner<>{}.value; }
  static_assert(f() == B, "");
};
Y<true> y;
Y<false> y2;
#endif

} // end ns1
} // end ns PR26134

namespace friends {
  namespace ns {
    template<typename> struct A {
      template<typename> friend void f();
      template<typename> friend struct X;
    };
    template<typename = int> void f(); // expected-warning 0-1{{extension}}
    template<typename = int> struct X;
    A<int> a;
  }
  namespace ns {
    void g() { f(); }
    X<int> *p;
  }
}

namespace unevaluated {
  int a;
  template<int = 0> int f(int = a); // expected-warning 0-1{{extension}}
  int k = sizeof(f());
}

#if __cplusplus >= 201103L
namespace GH68490 {

template <typename T> struct S {
  template <typename U>
  constexpr int SizeOfU(int param = sizeof(U)) const;

  template <typename U>
  constexpr int SizeOfT(int param = sizeof(T)) const;
};

template <typename T> struct S<T *> {
  template <typename U>
  constexpr int SizeOfU(int param = sizeof(U)) const;

  template <typename U>
  constexpr int SizeOfT(int param = sizeof(T *)) const;
};

template <typename T>
template <typename U>
constexpr int S<T *>::SizeOfU(int param) const {
  return param;
}

template <typename T>
template <typename U>
constexpr int S<T *>::SizeOfT(int param) const {
  return param;
}

template <>
template <typename T>
constexpr int S<int>::SizeOfU(int param) const {
  return param;
}

template <>
template <typename T>
constexpr int S<int>::SizeOfT(int param) const {
  return param;
}

static_assert(S<int>().SizeOfU<char>() == sizeof(char), "");
static_assert(S<int>().SizeOfT<char>() == sizeof(int), "");
static_assert(S<short *>().SizeOfU<char>() == sizeof(char), "");
static_assert(S<short *>().SizeOfT<char>() == sizeof(short *), "");

} // namespace GH68490

#endif
