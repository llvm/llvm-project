// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unused
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unused -fms-compatibility -DMSVC
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -Wno-unused
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -Wno-unused -fms-compatibility -DMSVC
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -Wno-unused
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -Wno-unused -fms-compatibility -DMSVC
namespace N {
  struct A {
    typedef int type;
  };

  struct B {
  };

  struct C {
    struct type { };
    int type; // expected-note 2{{referenced member 'type' is declared here}}
  };
}

int i;

typename N::A::type *ip1 = &i;
#if __cplusplus <= 199711L // C++03 or earlier modes
// expected-warning@-2 {{'typename' occurs outside of a template}}
#endif
typename N::B::type *ip2 = &i; // expected-error{{no type named 'type' in 'N::B'}}
#if __cplusplus <= 199711L
// expected-warning@-2 {{'typename' occurs outside of a template}}
#endif
typename N::C::type *ip3 = &i; // expected-error{{typename specifier refers to non-type member 'type'}}
#if __cplusplus <= 199711L
// expected-warning@-2 {{'typename' occurs outside of a template}}
#endif

void test(double d) {
  typename N::A::type f(typename N::A::type(a)); // expected-warning{{disambiguated as a function declaration}}
  // expected-note@-1 {{add a pair of parentheses}}
#if __cplusplus <= 199711L
  // expected-warning@-3 2{{'typename' occurs outside of a template}}
#endif
  int five = f(5);
  
  using namespace N;
  for (typename A::type i = 0; i < 10; ++i)
#if __cplusplus <= 199711L
// expected-warning@-2 {{'typename' occurs outside of a template}}
#endif
    five += 1;

  const typename N::A::type f2(d);
#if __cplusplus <= 199711L
// expected-warning@-2 {{'typename' occurs outside of a template}}
#endif
}

namespace N {
  template<typename T>
  struct X {
    typedef typename T::type type; // expected-error {{no type named 'type' in 'N::B'}} \
    // expected-error {{no type named 'type' in 'B'}} \
    // FIXME: location info for error above isn't very good \
    // expected-error 2{{typename specifier refers to non-type member 'type'}} \
    // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
  };
}

N::X<N::A>::type *ip4 = &i;
N::X<N::B>::type *ip5 = &i; // expected-note{{in instantiation of template class 'N::X<N::B>' requested here}}
N::X<N::C>::type *ip6 = &i; // expected-note{{in instantiation of template class 'N::X<N::C>' requested here}}

N::X<int>::type fail1; // expected-note{{in instantiation of template class 'N::X<int>' requested here}}

template<typename T>
struct Y {
  typedef typename N::X<T>::type *type; // expected-note{{in instantiation of template class 'N::X<B>' requested here}} \
  // expected-note{{in instantiation of template class 'N::X<C>' requested here}}
};

struct A {
  typedef int type;
};

struct B {
};

struct C {
  struct type { };
  int type; // expected-note{{referenced member 'type' is declared here}}
};

::Y<A>::type ip7 = &i;
::Y<B>::type ip8 = &i; // expected-note{{in instantiation of template class 'Y<B>' requested here}}
::Y<C>::type ip9 = &i; // expected-note{{in instantiation of template class 'Y<C>' requested here}}

template<typename T> struct D {
  typedef typename T::foo foo;  // expected-error {{type 'long' cannot be used prior to '::' because it has no members}}
  typedef typename foo::bar bar;
};

D<long> struct_D;  // expected-note {{in instantiation of template class 'D<long>' requested here}}

template<typename T> struct E {
  typedef typename T::foo foo;
  typedef typename foo::bar bar;  // expected-error {{type 'E<F>::foo' (aka 'double') cannot be used prior to '::' because it has no members}}
};

struct F {
  typedef double foo;
};

E<F> struct_E; // expected-note {{in instantiation of template class 'E<F>' requested here}}

template<typename T> struct G {
  typedef typename T::foo foo;
  typedef typename foo::bar bar;
};

struct H {
  struct foo {
    typedef double bar;
  };
};

G<H> struct_G;

namespace PR10925 {
  template< int mydim, typename Traits >
  class BasicGeometry
  {
    typedef int some_type_t;
  };

  template<class ctype, int mydim, int coorddim>
  class MockGeometry : BasicGeometry<mydim, int>{
    using typename BasicGeometry<mydim, int>::operator[]; // expected-error {{typename is allowed for identifiers only}}
  };
}


namespace missing_typename {
template <class T1, class T2> struct pair {}; // expected-note 7 {{template parameter is declared here}}

template <class T1, class T2>
struct map {
  typedef T1* iterator;
};

template <class T>
class ExampleClass1 {
  struct ExampleItem;


  struct ExampleItemSet {
    typedef ExampleItem* iterator;
    ExampleItem* operator[](unsigned);
  };

  void foo() {
#ifdef MSVC
    // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
    // expected-error@+2 {{template argument for template type parameter must be a type; did you forget 'typename'?}}
#endif
    pair<ExampleItemSet::iterator, int> i;
    pair<this->ExampleItemSet::iterator, int> i; // expected-error-re {{template argument for template type parameter must be a type{{$}}}}
    pair<ExampleItemSet::operator[], int> i; // expected-error-re {{template argument for template type parameter must be a type{{$}}}}
  }
#ifdef MSVC
    // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
  // expected-error@+2 {{template argument for template type parameter must be a type; did you forget 'typename'?}}
#endif
  pair<ExampleItemSet::iterator, int> elt;


  typedef map<int, ExampleItem*> ExampleItemMap;

  static void bar() {
#ifdef MSVC
    // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
    // expected-error@+2 {{template argument for template type parameter must be a type; did you forget 'typename'?}}
#endif
    pair<ExampleItemMap::iterator, int> i;
  }
#ifdef MSVC
    // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
  // expected-error@+2 {{template argument for template type parameter must be a type; did you forget 'typename'?}}
#endif
  pair<ExampleItemMap::iterator, int> entry;
  pair<bar, int> foobar; // expected-error {{template argument for template type parameter must be a type}}
};
} // namespace missing_typename

namespace missing_typename_and_base {
template <class T> struct Bar {}; // expected-note 1+ {{template parameter is declared here}}
template <typename T>
struct Foo : T {

  // FIXME: MSVC accepts this code.
  Bar<TypeInBase> x; // expected-error {{use of undeclared identifier 'TypeInBase'}}

#ifdef MSVC
  // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
  // expected-error@+2 {{must be a type; did you forget 'typename'?}}
#endif
  Bar<T::TypeInBase> y;

#ifdef MSVC
  // expected-warning@+4 {{omitted 'typename' is a Microsoft extension}}
#else
  // expected-error@+2 {{must be a type; did you forget 'typename'?}}
#endif
  Bar<T::NestedRD::TypeInNestedRD> z;

};
struct Base {
  typedef int TypeInBase;
  struct NestedRD {
    typedef int TypeInNestedRD;
  };
};
Foo<Base> x;
} // namespace missing_typename_and_base

namespace func_type_vs_construct_tmp {
template <typename> struct S { typedef int type; };
template <typename T> void f();
template <int N> void f();

// expected-error@+1 {{missing 'typename' prior to dependent type name 'S<int>::type'}}
template <typename T> void g() { f</*typename*/ S<T>::type(int())>(); }

// Adding typename does fix the diagnostic.
template <typename T> void h() { f<typename S<T>::type(int())>(); }

void j() {
  g<int>(); // expected-note-re {{in instantiation {{.*}} requested here}}
  h<int>();
}
} // namespace func_type_vs_construct_tmp

namespace pointer_vs_multiply {
int x;
// expected-error@+1 {{missing 'typename' prior to dependent type name 'B::type_or_int'}}
template <typename T> void g() { T::type_or_int * x; }
// expected-error@+1 {{typename specifier refers to non-type member 'type_or_int' in 'pointer_vs_multiply::A'}}
template <typename T> void h() { typename T::type_or_int * x; }

struct A { static const int type_or_int = 5; }; // expected-note {{referenced member 'type_or_int' is declared here}}
struct B { typedef int type_or_int; };

void j() {
  g<A>();
  g<B>(); // expected-note-re {{in instantiation {{.*}} requested here}}
  h<A>(); // expected-note-re {{in instantiation {{.*}} requested here}}
  h<B>();
}
} // namespace pointer_vs_multiply
