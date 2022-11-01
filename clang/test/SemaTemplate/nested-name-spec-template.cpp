// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace N { 
  namespace M {
    template<typename T> struct Promote;
    
    template<> struct Promote<short> {
      typedef int type;
    };
    
    template<> struct Promote<int> {
      typedef int type;
    };
    
    template<> struct Promote<float> {
      typedef double type;
    };
    
    Promote<short>::type *ret_intptr(int* ip) { return ip; }
    Promote<int>::type *ret_intptr2(int* ip) { return ip; }
  }

  M::Promote<int>::type *ret_intptr3(int* ip) { return ip; }
  M::template Promote<int>::type *ret_intptr4(int* ip) { return ip; } 
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'template' keyword outside of a template}}
#endif

  M::template Promote<int> pi;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'template' keyword outside of a template}}
#endif
}

N::M::Promote<int>::type *ret_intptr5(int* ip) { return ip; }
::N::M::Promote<int>::type *ret_intptr6(int* ip) { return ip; }


N::M::template; // expected-error{{expected unqualified-id}}
N::M::template Promote; // expected-error{{expected unqualified-id}}

namespace N {
  template<typename T> struct A;

  template<>
  struct A<int> {
    struct X;
  };

  struct B; // expected-note{{declared as a non-template here}}
}

struct ::N::A<int>::X {
  int foo;
};

template<typename T>
struct TestA {
  typedef typename N::template B<T>::type type; // expected-error{{'B' following the 'template' keyword does not refer to a template}}
};

// Reduced from a Boost failure.
namespace test1 {
  template <class T> struct pair {
    T x;
    T y;

    static T pair<T>::* const mem_array[2];
  };

  template <class T>
  T pair<T>::* const pair<T>::mem_array[2] = { &pair<T>::x, &pair<T>::y };
}

typedef int T;
namespace N1 {
  template<typename T> T f0();
}

template<typename T> T N1::f0() { }

namespace PR7385 {
  template< typename > struct has_xxx0
  {
    template< typename > struct has_xxx0_introspect
    {
      template< typename > struct has_xxx0_substitute ;
      template< typename V > 
      int int00( has_xxx0_substitute < typename V::template xxx< > > = 0 );
    };
    static const int value = has_xxx0_introspect<int>::value; // expected-error{{no member named 'value'}}
    typedef int type;
  };

  has_xxx0<int>::type t; // expected-note{{instantiation of}}
}

namespace PR7725 {
  template<class ignored> struct TypedefProvider;
  template<typename T>
  struct TemplateClass : public TypedefProvider<T>
  {
    void PrintSelf() {
      TemplateClass::Test::PrintSelf();
    }
  };
}

namespace PR9226 {
  template<typename a>
  void nt() // expected-note{{function template 'nt' declared here}}
  { nt<>:: } // expected-error{{qualified name refers into a specialization of function template 'nt'}} \
  // expected-error{{expected unqualified-id}}

  template<typename T>
  void f(T*); // expected-note{{function template 'f' declared here}}

  template<typename T>
  void f(T*, T*); // expected-note{{function template 'f' declared here}}

  void g() {
    f<int>:: // expected-error{{qualified name refers into a specialization of function template 'f'}}
  } // expected-error{{expected unqualified-id}}

  struct X {
    template<typename T> void f(); // expected-note{{function template 'f' declared here}}
  };

  template<typename T, typename U>
  struct Y {
    typedef typename T::template f<U> type; // expected-error{{template name refers to non-type template 'X::template f'}}
  };

  Y<X, int> yxi; // expected-note{{in instantiation of template class 'PR9226::Y<PR9226::X, int>' requested here}}
}

namespace PR9449 {
  template <typename T>
  struct s; // expected-note{{template is declared here}}

  template <typename T>
  void f() {
    int s<T>::template n<T>::* f; // expected-error{{implicit instantiation of undefined template 'PR9449::s<int>'}}
  }

  template void f<int>(); // expected-note{{in instantiation of}}
}

namespace sugared_template_instantiation {
  // Test that we ignore local qualifiers.
  template <class A1, class = typename A1::type1> struct A {};
  struct B { typedef int type1; };
  typedef A<const B> type2;
} // namespace sugated_template_instantiation
