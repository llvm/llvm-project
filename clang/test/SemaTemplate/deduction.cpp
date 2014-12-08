// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

// Template argument deduction with template template parameters.
template<typename T, template<T> class A> 
struct X0 {
  static const unsigned value = 0;
};

template<template<int> class A>
struct X0<int, A> {
  static const unsigned value = 1;
};

template<int> struct X0i;
template<long> struct X0l;
int array_x0a[X0<long, X0l>::value == 0? 1 : -1];
int array_x0b[X0<int, X0i>::value == 1? 1 : -1];

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename T> struct allocator { };
template<typename T, typename Alloc = allocator<T> > struct vector {};

// Fun with meta-lambdas!
struct _1 {};
struct _2 {};

// Replaces all occurrences of _1 with Arg1 and _2 with Arg2 in T.
template<typename T, typename Arg1, typename Arg2>
struct Replace {
  typedef T type;
};

// Replacement of the whole type.
template<typename Arg1, typename Arg2>
struct Replace<_1, Arg1, Arg2> {
  typedef Arg1 type;
};

template<typename Arg1, typename Arg2>
struct Replace<_2, Arg1, Arg2> {
  typedef Arg2 type;
};

// Replacement through cv-qualifiers
template<typename T, typename Arg1, typename Arg2>
struct Replace<const T, Arg1, Arg2> {
  typedef typename Replace<T, Arg1, Arg2>::type const type;
};

// Replacement of templates
template<template<typename> class TT, typename T1, typename Arg1, typename Arg2>
struct Replace<TT<T1>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type> type;
};

template<template<typename, typename> class TT, typename T1, typename T2,
         typename Arg1, typename Arg2>
struct Replace<TT<T1, T2>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type,
             typename Replace<T2, Arg1, Arg2>::type> type;
};

// Just for kicks...
template<template<typename, typename> class TT, typename T1,
         typename Arg1, typename Arg2>
struct Replace<TT<T1, _2>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type, Arg2> type;
};

int array0[is_same<Replace<_1, int, float>::type, int>::value? 1 : -1];
int array1[is_same<Replace<const _1, int, float>::type, const int>::value? 1 : -1];
int array2[is_same<Replace<vector<_1>, int, float>::type, vector<int> >::value? 1 : -1];
int array3[is_same<Replace<vector<const _1>, int, float>::type, vector<const int> >::value? 1 : -1];
int array4[is_same<Replace<vector<int, _2>, double, float>::type, vector<int, float> >::value? 1 : -1];

// PR5911
template <typename T, int N> void f(const T (&a)[N]);
int iarr[] = { 1 };
void test_PR5911() { f(iarr); }

// Must not examine base classes of incomplete type during template argument
// deduction.
namespace PR6257 {
  template <typename T> struct X {
    template <typename U> X(const X<U>& u);
  };
  struct A;
  void f(A& a);
  void f(const X<A>& a);
  void test(A& a) { (void)f(a); }
}

// PR7463
namespace PR7463 {
  const int f ();
  template <typename T_> void g (T_&); // expected-note{{T_ = int}}
  void h (void) { g(f()); } // expected-error{{no matching function for call}}
}

namespace test0 {
  template <class T> void make(const T *(*fn)()); // expected-note {{candidate template ignored: can't deduce a type for 'T' which would make 'const T' equal 'char'}}
  char *char_maker();
  void test() {
    make(char_maker); // expected-error {{no matching function for call to 'make'}}
  }
}

namespace test1 {
  template<typename T> void foo(const T a[3][3]);
  void test() {
    int a[3][3];
    foo(a);
  }
}

// PR7708
namespace test2 {
  template<typename T> struct Const { typedef void const type; };

  template<typename T> void f(T, typename Const<T>::type*);
  template<typename T> void f(T, void const *);

  void test() {
    void *p = 0;
    f(0, p);
  }
}

// rdar://problem/8537391
namespace test3 {
  struct Foo {
    template <void F(char)> static inline void foo();
  };

  class Bar {
    template<typename T> static inline void wobble(T ch);

  public:
    static void madness() {
      Foo::foo<wobble<char> >();
    }
  };
}

// Verify that we can deduce enum-typed arguments correctly.
namespace test14 {
  enum E { E0, E1 };
  template <E> struct A {};
  template <E e> void foo(const A<e> &a) {}

  void test() {
    A<E0> a;
    foo(a);
  }
}

namespace PR21536 {
  template<typename ...T> struct X;
  template<typename A, typename ...B> struct S {
    static_assert(sizeof...(B) == 1, "");
    void f() {
      using T = A;
      using T = int;

      using U = X<B...>;
      using U = X<int>;
    }
  };
  template<typename ...T> void f(S<T...>);
  void g() { f(S<int, int>()); }
}

namespace PR19372 {
  template <template<typename...> class C, typename ...Us> struct BindBack {
    template <typename ...Ts> using apply = C<Ts..., Us...>;
  };
  template <typename, typename...> struct Y;
  template <typename ...Ts> using Z = Y<Ts...>;

  using T = BindBack<Z, int>::apply<>;
  using T = Z<int>;

  using U = BindBack<Z, int, int>::apply<char>;
  using U = Z<char, int, int>;
}
