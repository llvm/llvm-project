// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wconversion -verify %s
template<int N> struct A; // expected-note 5{{template parameter is declared here}}

A<0> *a0;

A<int()> *a1; // expected-error{{template argument for non-type template parameter is treated as function type 'int ()'}}

A<int> *a2; // expected-error{{template argument for non-type template parameter must be an expression}}

A<1 >> 2> *a3; // expected-warning{{use of right-shift operator ('>>') in template argument will require parentheses in C++11}}

// C++ [temp.arg.nontype]p5:
A<A> *a4; // expected-error{{must be an expression}}

enum E { Enumerator = 17 };
A<E> *a5; // expected-error{{template argument for non-type template parameter must be an expression}}
template<E Value> struct A1; // expected-note{{template parameter is declared here}}
A1<Enumerator> *a6; // okay
A1<17> *a7; // expected-error{{non-type template argument of type 'int' cannot be converted to a value of type 'E'}}

const long LongValue = 12345678;
A<LongValue> *a8;
const short ShortValue = 17;
A<ShortValue> *a9;

int f(int);
A<f(17)> *a10; // expected-error{{non-type template argument of type 'int' is not an integral constant expression}}

class X {
public:
  X();
  X(int, int);
  operator int() const;
};
A<X(17, 42)> *a11; // expected-error{{non-type template argument of type 'X' must have an integral or enumeration type}}

float f(float);

float g(float); // expected-note 2{{candidate function}}
double g(double); // expected-note 2{{candidate function}}

int h(int);
float h2(float);

template<int fp(int)> struct A3; // expected-note 1{{template parameter is declared here}}
A3<h> *a14_1;
A3<&h> *a14_2;
A3<f> *a14_3;
A3<&f> *a14_4;
A3<h2> *a14_6;  // expected-error{{non-type template argument of type 'float (float)' cannot be converted to a value of type 'int (*)(int)'}}
A3<g> *a14_7; // expected-error{{address of overloaded function 'g' does not match required type 'int (int)'}}


struct Y { } y;

volatile X * X_volatile_ptr;
template<X const &AnX> struct A4; // expected-note 2{{template parameter is declared here}}
X an_X;
A4<an_X> *a15_1; // okay
A4<*X_volatile_ptr> *a15_2; // expected-error{{non-type template argument does not refer to any declaration}}
A4<y> *15_3; //  expected-error{{non-type template parameter of reference type 'const X &' cannot bind to template argument of type 'struct Y'}} \
            // FIXME: expected-error{{expected unqualified-id}}

template<int (&fr)(int)> struct A5; // expected-note{{template parameter is declared here}}
A5<h> *a16_1;
A5<f> *a16_3;
A5<h2> *a16_6;  // expected-error{{non-type template parameter of reference type 'int (&)(int)' cannot bind to template argument of type 'float (float)'}}
A5<g> *a14_7; // expected-error{{address of overloaded function 'g' does not match required type 'int (int)'}}

struct Z {
  int foo(int);
  float bar(float);
  int bar(int);
  double baz(double);

  int int_member;
  float float_member;
  union {
    int union_member;
  };
};
template<int (Z::*pmf)(int)> struct A6; // expected-note{{template parameter is declared here}}
A6<&Z::foo> *a17_1;
A6<&Z::bar> *a17_2;
A6<&Z::baz> *a17_3; // expected-error-re{{non-type template argument of type 'double (Z::*)(double){{( __attribute__\(\(thiscall\)\))?}}' cannot be converted to a value of type 'int (Z::*)(int){{( __attribute__\(\(thiscall\)\))?}}'}}


template<int Z::*pm> struct A7;  // expected-note{{template parameter is declared here}}
template<int Z::*pm> struct A7c;
A7<&Z::int_member> *a18_1;
A7c<&Z::int_member> *a18_2;
A7<&Z::float_member> *a18_3; // expected-error{{non-type template argument of type 'float Z::*' cannot be converted to a value of type 'int Z::*'}}
A7c<(&Z::int_member)> *a18_4; // expected-warning{{address non-type template argument cannot be surrounded by parentheses}}
A7c<&Z::union_member> *a18_5;

template<unsigned char C> struct Overflow; // expected-note{{template parameter is declared here}}

Overflow<5> *overflow1; // okay
Overflow<255> *overflow2; // okay
Overflow<256> *overflow3; // expected-warning{{non-type template argument value '256' truncated to '0' for template parameter of type 'unsigned char'}}


template<unsigned> struct Signedness; // expected-note{{template parameter is declared here}}
Signedness<10> *signedness1; // okay
Signedness<-10> *signedness2; // expected-warning{{non-type template argument with value '-10' converted to '4294967286' for unsigned template parameter of type 'unsigned int'}}

template<signed char C> struct SignedOverflow; // expected-note 3 {{template parameter is declared here}}
SignedOverflow<1> *signedoverflow1;
SignedOverflow<-1> *signedoverflow2;
SignedOverflow<-128> *signedoverflow3;
SignedOverflow<-129> *signedoverflow4; // expected-warning{{non-type template argument value '-129' truncated to '127' for template parameter of type 'signed char'}}
SignedOverflow<127> *signedoverflow5;
SignedOverflow<128> *signedoverflow6; // expected-warning{{non-type template argument value '128' truncated to '-128' for template parameter of type 'signed char'}}
SignedOverflow<(unsigned char)128> *signedoverflow7; // expected-warning{{non-type template argument value '128' truncated to '-128' for template parameter of type 'signed char'}}

// Check canonicalization of template arguments.
template<int (*)(int, int)> struct FuncPtr0;
int func0(int, int);
extern FuncPtr0<&func0> *fp0;
template<int (*)(int, int)> struct FuncPtr0;
extern FuncPtr0<&func0> *fp0;
int func0(int, int);
extern FuncPtr0<&func0> *fp0;

// PR5350
namespace ns {
  template <typename T>
  struct Foo {
    static const bool value = true;
  };

  template <bool b>
  struct Bar {};

  const bool value = false;

  Bar<bool(ns::Foo<int>::value)> x;
}

// PR5349
namespace ns {
  enum E { k };

  template <E e>
  struct Baz  {};

  Baz<k> f1;  // This works.
  Baz<E(0)> f2;  // This too.
  Baz<static_cast<E>(0)> f3;  // And this.

  Baz<ns::E(0)> b1;  // This doesn't work.
  Baz<static_cast<ns::E>(0)> b2;  // This neither.
}

// PR5597
template<int (*)(float)> struct X0 { };

struct X1 {
    static int pfunc(float);
};
void test_X0_X1() {
  X0<X1::pfunc> x01;
}

// PR6249
namespace pr6249 {
  template<typename T, T (*func)()> T f() {
    return func();
  }

  int h();
  template int f<int, h>();
}

namespace PR6723 {
  template<unsigned char C> void f(int (&a)[C]); // expected-note 3{{candidate template ignored: substitution failure [with C = '\x00']}}
  // expected-note@-1 {{not viable: no known conversion from 'int[512]' to 'int (&)[0]'}}
  void g() {
    int arr512[512];
    f(arr512); // expected-error{{no matching function for call}}
    f<512>(arr512); // expected-error{{no matching function for call}}

    int arr0[0];
    f(arr0); // expected-error{{no matching function for call}}
    f<0>(arr0); // expected-error{{no matching function for call}}
  }
}

// Check that we instantiate declarations whose addresses are taken
// for non-type template arguments.
namespace EntityReferenced {
  template<typename T, void (*)(T)> struct X { };

  template<typename T>
  struct Y {
    static void f(T x) {
      x = 1; // expected-error{{incompatible integer to pointer conversion assigning to 'int *' from 'int'}}
    }
  };

  void g() {
    typedef X<int*, Y<int*>::f> x; // expected-note{{in instantiation of}}
  }
}

namespace PR6964 {
  template <typename ,int, int = 9223372036854775807L > // expected-warning {{non-type template argument value '9223372036854775807' truncated to '-1' for template parameter of type 'int'}} \
  // expected-note {{template parameter is declared here}}
  struct as_nview { };

  template <typename Sequence, int I0>
  struct as_nview<Sequence, I0>  // expected-note{{while checking a default template argument used here}}
  { };
}

namespace test8 {
  template <int* ip> struct A {
    int* p;
    A() : p(ip) {}
  };

  void test0() {
    extern int i00;
    A<&i00> a00;
  }

  extern int i01;
  void test1() {
    A<&i01> a01;
  }


  struct C {
    int x;
    char y;
    double z;
  };

  template <C* cp> struct B {
    C* p;
    B() : p(cp) {}
  };

  void test2() {
    extern C c02;
    B<&c02> b02;
  }

  extern C c03;
  void test3() {
    B<&c03> b03;
  }
}

namespace PR8372 {
  template <int I> void foo() { } // expected-note{{template parameter is declared here}}
  void bar() { foo <0x80000000> (); } // expected-warning{{non-type template argument value '2147483648' truncated to '-2147483648' for template parameter of type 'int'}}
}

namespace PR9227 {
  template <bool B> struct enable_if_bool { };
  template <> struct enable_if_bool<true> { typedef int type; }; // expected-note{{'enable_if_bool<true>::type' declared here}}
  void test_bool() { enable_if_bool<false>::type i; } // expected-error{{enable_if_bool<false>'; did you mean 'enable_if_bool<true>::type'?}}

  template <char C> struct enable_if_char { };
  template <> struct enable_if_char<'a'> { typedef int type; }; // expected-note 5{{'enable_if_char<'a'>::type' declared here}}
  void test_char_0() { enable_if_char<0>::type i; } // expected-error{{enable_if_char<'\x00'>'; did you mean 'enable_if_char<'a'>::type'?}}
  void test_char_b() { enable_if_char<'b'>::type i; } // expected-error{{enable_if_char<'b'>'; did you mean 'enable_if_char<'a'>::type'?}}
  void test_char_possibly_negative() { enable_if_char<'\x02'>::type i; } // expected-error{{enable_if_char<'\x02'>'; did you mean 'enable_if_char<'a'>::type'?}}
  void test_char_single_quote() { enable_if_char<'\''>::type i; } // expected-error{{enable_if_char<'\''>'; did you mean 'enable_if_char<'a'>::type'?}}
  void test_char_backslash() { enable_if_char<'\\'>::type i; } // expected-error{{enable_if_char<'\\'>'; did you mean 'enable_if_char<'a'>::type'?}}

  template <int N> struct enable_if_int {};
  template <> struct enable_if_int<1> { typedef int type; }; // expected-note{{'enable_if_int<1>::type' declared here}}
  void test_int() { enable_if_int<2>::type i; } // expected-error{{enable_if_int<2>'; did you mean 'enable_if_int<1>::type'?}}

  template <unsigned int N> struct enable_if_unsigned_int {};
  template <> struct enable_if_unsigned_int<1> { typedef int type; }; // expected-note{{'enable_if_unsigned_int<1>::type' declared here}}
  void test_unsigned_int() { enable_if_unsigned_int<2>::type i; } // expected-error{{enable_if_unsigned_int<2>'; did you mean 'enable_if_unsigned_int<1>::type'?}}

  template <unsigned long long N> struct enable_if_unsigned_long_long {};
  template <> struct enable_if_unsigned_long_long<1> { typedef int type; }; // expected-note{{'enable_if_unsigned_long_long<1>::type' declared here}}
  void test_unsigned_long_long() { enable_if_unsigned_long_long<2>::type i; } // expected-error{{enable_if_unsigned_long_long<2>'; did you mean 'enable_if_unsigned_long_long<1>::type'?}}

  template <long long N> struct enable_if_long_long {};
  template <> struct enable_if_long_long<1> { typedef int type; }; // expected-note{{'enable_if_long_long<1>::type' declared here}}
  void test_long_long() { enable_if_long_long<2>::type i; } // expected-error{{enable_if_long_long<2>'; did you mean 'enable_if_long_long<1>::type'?}}

}

namespace PR10579 {
  namespace fcppt
  {
    namespace container
    {
      namespace bitfield
      {

        template<
          typename Enum,
          Enum Size
          >
        class basic;

        template<
          typename Enum,
          Enum Size
          >
        class basic
        {
        public:
          basic()
          {
          }
        };

      }
    }
  }

  namespace
  {

    namespace testenum
    {
      enum type
        {
          foo,
          bar,
          size
        };
    }

  }

  int main()
  {
    typedef fcppt::container::bitfield::basic<
    testenum::type,
      testenum::size
      > bitfield_foo;

    bitfield_foo obj;
  }

}

template <int& I> struct PR10766 { static int *ip; };
template <int& I> int* PR10766<I>::ip = &I;

namespace rdar13000548 {
  template<typename R, typename U, R F>
  U f() { return &F; } // expected-error{{cannot take the address of an rvalue of type 'int (*)(int)'}} expected-error{{cannot take the address of an rvalue of type 'int *'}}

  int g(int);
  int y[3];
  void test()
  {
    f<int(int), int (*)(int), g>(); // expected-note{{in instantiation of}}
    f<int[3], int*, y>(); // expected-note{{in instantiation of}}
  }

}

namespace rdar13806270 {
  template <unsigned N> class X { };
  const unsigned value = 32;
  struct Y {
    X<value + 1> x;
  };
  void foo() {}
}

namespace PR17696 {
  struct a {
    union {
      int i;
    };
  };

  template <int (a::*p)> struct b : a {
    b() { this->*p = 0; }
  };

  b<&a::i> c; // okay
}

namespace partial_order_different_types {
  template<int, int, typename T, typename, T> struct A;
  template<int N, typename T, typename U, T V> struct A<0, N, T, U, V>; // expected-note {{matches}}
  // FIXME: It appears that this partial specialization should be ill-formed as
  // it is not more specialized than the primary template. V is not deducible
  // because it does not have the same type as the corresponding parameter.
  template<int N, typename T, typename U, U V> struct A<0, N, T, U, V> {}; // expected-note {{matches}}
  A<0, 0, int, int, 0> a; // expected-error {{ambiguous}}
}

namespace partial_order_references {
  // FIXME: The standard does not appear to consider the second specialization
  // to be more specialized than the first! The problem is that deducing
  // an 'int&' parameter from an argument 'R' results in a type mismatch,
  // because the parameter has a reference type and the argument is an
  // expression and thus does not have reference type. We resolve this by
  // matching the type of an expression corresponding to the parameter rather
  // than matching the parameter itself.
  template <int, int, int &> struct A {};
  template <int N, int &R> struct A<N, 0, R> {};
  template <int &R> struct A<0, 0, R> {};
  int N;
  A<0, 0, N> a;

  template<int, int &R> struct B; // expected-note 2{{template}}
  template<const int &R> struct B<0, R> {};
  // expected-error@-1 {{not more specialized than the primary}}
  // expected-note@-2 {{'const int' vs 'int &'}}
  B<0, N> b; // expected-error {{undefined}}

  template<int, const int &R> struct C; // expected-note 2{{template}}
  template<int &R> struct C<0, R> {};
  // expected-error@-1 {{not more specialized than the primary}}
  // expected-note@-2 {{'int' vs 'const int &'}}
  C<0, N> c; // expected-error {{undefined}}

  template<int, const int &R> struct D; // expected-note 2{{template}}
  template<int N> struct D<0, N> {};
  // expected-error@-1 {{not more specialized than the primary}}
  // expected-note@-2 {{'int' vs 'const int &'}}
  extern const int K = 5;
  D<0, K> d; // expected-error {{undefined}}
}

namespace dependent_nested_partial_specialization {
  template<typename> using X = int; // expected-warning {{C++11}}
  template<typename T> using Y = T*; // expected-warning {{C++11}}
  int n;

  template<template<typename> class X> struct A {
    template<typename T, X<T> N> struct B; // expected-note 2{{here}}
    template <typename T> struct B<T, 0> {}; // expected-error {{non-type template argument specializes a template parameter with dependent type 'Y<T>' (aka 'T *')}}
  };
  A<X>::B<int, 0> ax;
  A<Y>::B<int, &n> ay; // expected-error {{undefined}} expected-note {{instantiation of}}

  template<template<typename> class X> struct C {
    template<typename T, int N, int M> struct D; // expected-note {{here}}
    template<typename T, X<T> N> struct D<T*, N, N + 1> {}; // expected-error {{type of specialized non-type template argument depends on}}
  };
  C<X>::D<int*, 0, 1> cx;
  C<Y>::D<int*, 0, 1> cy; // expected-error {{undefined}} expected-note {{instantiation of}}

  template<typename T> struct E {
    template<typename U, U V> struct F; // expected-note {{template}}
    template<typename W, T V> struct F<W, V> {}; // expected-error {{not more specialized than the primary}}
  };
  E<int>::F<int, 0> e1; // expected-note {{instantiation of}}
}

namespace nondependent_default_arg_ordering {
  int n, m;
  template<typename A, A B = &n> struct X {};
  template<typename A> void f(X<A>);
  template<typename A> void f(X<A, &m>);
  template<typename A, A B> void f(X<A, B>);
  template<template<typename U, U> class T, typename A, int *B> void f(T<A, B>);
  void g() {
    X<int *, &n> x; f(x);
    X<int *, &m> y; f(y);
  }
}

namespace pointer_to_char_array {
  typedef char T[4];
  template<T *P> struct A { void f(); };
  template<T *P> void A<P>::f() {}
  T foo = "foo";
  void g() { A<&foo>().f(); }
}

namespace dependent_backreference {
  struct Incomplete; // expected-note 2{{forward declaration}}
  Incomplete f(int); // expected-note 2{{here}}
  int f(short);

  template<typename T, T Value, int(*)[sizeof(f(Value))]> struct X {}; // expected-error 2{{incomplete}}
  int arr[sizeof(int)];
  // When checking this template-id, we must not treat 'Value' as having type
  // 'int'; its type is the dependent type 'T'.
  template<typename T> void f() { X<T, 0, &arr> x; } // expected-note {{substituting}}
  void g() { f<short>(); }
  void h() { f<int>(); } // expected-note {{instantiation}}

  // The second of these is OK to diagnose eagerly because 'Value' has the
  // non-dependent type 'int'.
  template<short S> void a() { X<short, S, &arr> x; }
  template<short S> void b() { X<int, S, &arr> x; } // expected-note {{substituting}}
}

namespace instantiation_dependent {
  template<typename T, __typeof(sizeof(T))> void f(int);
  template<typename T, __typeof(sizeof(0))> int &f(...);
  int &rf = f<struct incomplete, 0>(0);

  int arr[sizeof(sizeof(int))];
  template<typename T, int (*)[sizeof(sizeof(T))]> void g(int);
  template<typename T, int (*)[sizeof(sizeof(int))]> int &g(...);
  int &rg = g<struct incomplete, &arr>(0);
}

namespace complete_array_from_incomplete {
  template <typename T, const char* const A[static_cast<int>(T::kNum)]>
  class Base {};
  template <class T, const char* const A[]>
  class Derived : public Base<T, A> {};

  struct T {
    static const int kNum = 3;
  };
  extern const char *const kStrs[3] = {};
  Derived<T, kStrs> d;
}

namespace type_of_pack {
  template<typename ...T> struct A { // expected-warning 0-1{{extension}}
    template<T *...V> void f() {
      g(V.f() ...); // expected-error {{base type 'T *' is not a structure or union}}
    }
  };
}

namespace match_type_after_substitution {
  template<typename T> struct X {};
  X<int> y;
  template<typename T, X<T> &Y> struct B {
    typedef B<T, Y> Self;
  };

  // These two formulations should resolve to the same type.
  typedef B<int, y> Z;
  typedef Z::Self Z;
}
