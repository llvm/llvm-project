// RUN: %clang_cc1 -fsyntax-only -verify %s
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

typename N::A::type *ip1 = &i; // expected-warning{{'typename' occurs outside of a template}}
typename N::B::type *ip2 = &i; // expected-error{{no type named 'type' in 'N::B'}} \
// expected-warning{{'typename' occurs outside of a template}}
typename N::C::type *ip3 = &i; // expected-error{{typename specifier refers to non-type member 'type'}} \
// expected-warning{{'typename' occurs outside of a template}}

void test(double d) {
  typename N::A::type f(typename N::A::type(a)); // expected-warning{{disambiguated as a function declaration}} \
  // expected-note{{add a pair of parentheses}} expected-warning 2{{'typename' occurs outside of a template}}
  int five = f(5);
  
  using namespace N;
  for (typename A::type i = 0; i < 10; ++i) // expected-warning{{'typename' occurs outside of a template}}
    five += 1;

  const typename N::A::type f2(d); // expected-warning{{'typename' occurs outside of a template}}
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
  typedef typename foo::bar bar;  // expected-error {{type 'foo' (aka 'double') cannot be used prior to '::' because it has no members}}
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
    pair<ExampleItemSet::iterator, int> i; // expected-error {{template argument for template type parameter must be a type; did you forget 'typename'?}}
    pair<this->ExampleItemSet::iterator, int> i; // expected-error-re {{template argument for template type parameter must be a type$}}
    pair<ExampleItemSet::operator[], int> i; // expected-error-re {{template argument for template type parameter must be a type$}}
  }
  pair<ExampleItemSet::iterator, int> elt; // expected-error {{template argument for template type parameter must be a type; did you forget 'typename'?}}


  typedef map<int, ExampleItem*> ExampleItemMap;

  static void bar() {
    pair<ExampleItemMap::iterator, int> i; // expected-error {{template argument for template type parameter must be a type; did you forget 'typename'?}}
  }
  pair<ExampleItemMap::iterator, int> entry; // expected-error {{template argument for template type parameter must be a type; did you forget 'typename'?}}
  pair<bar, int> foobar; // expected-error {{template argument for template type parameter must be a type}}
};
} // namespace missing_typename
