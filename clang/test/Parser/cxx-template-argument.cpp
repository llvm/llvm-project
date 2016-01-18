// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -fdelayed-template-parsing

template<typename T> struct A {};

// Check for template argument lists followed by junk
// FIXME: The diagnostics here aren't great...
A<int+> int x; // expected-error {{expected '>'}} expected-error {{expected unqualified-id}}
A<int x; // expected-error {{type-id cannot have a name}} expected-error {{expected '>'}}

// PR8912
template <bool> struct S {};
S<bool(2 > 1)> s;

// Test behavior when a template-id is ended by a token which starts with '>'.
namespace greatergreater {
  template<typename T> struct S { S(); S(T); };
  void f(S<int>=0); // expected-error {{a space is required between a right angle bracket and an equals sign (use '> =')}}
  void f(S<S<int>>=S<int>()); // expected-error {{use '> >'}} expected-error {{use '> ='}}
  template<typename T> void t();
  void g() {
    void (*p)() = &t<int>;
    (void)(&t<int>==p); // expected-error {{use '> ='}}
    (void)(&t<int>>=p); // expected-error {{use '> >'}}
    (void)(&t<S<int>>>=p); // expected-error {{use '> >'}}
    (void)(&t<S<int>>==p); // expected-error {{use '> >'}} expected-error {{use '> ='}}
  }
}

namespace PR5925 {
  template <typename x>
  class foo { // expected-note {{here}}
  };
  void bar(foo *X) { // expected-error {{requires template arguments}}
  }
}

namespace PR13210 {
  template <class T>
  class C {}; // expected-note {{here}}

  void f() {
    new C(); // expected-error {{requires template arguments}}
  }
}

// Don't emit spurious messages
namespace pr16225add {

  template<class T1, typename T2> struct Known { }; // expected-note 3 {{template is declared here}}
  template<class T1, typename T2> struct X;
  template<class T1, typename T2> struct ABC; // expected-note {{template is declared here}}
  template<int N1, int N2> struct ABC2 {};

  template<class T1, typename T2> struct foo :
    UnknownBase<T1,T2> // expected-error {{unknown template name 'UnknownBase'}}
  { };

  template<class T1, typename T2> struct foo2 :
    UnknownBase<T1,T2>, // expected-error {{unknown template name 'UnknownBase'}}
    Known<T1>  // expected-error {{too few template arguments for class template 'Known'}}
  { };

  template<class T1, typename T2> struct foo3 :
    UnknownBase<T1,T2,ABC<T2,T1> > // expected-error {{unknown template name 'UnknownBase'}}
  { };

  template<class T1, typename T2> struct foo4 :
    UnknownBase<T1,ABC<T2> >, // expected-error {{unknown template name 'UnknownBase'}} \
                              // expected-error {{too few template arguments for class template 'ABC'}}
    Known<T1>  // expected-error {{too few template arguments for class template 'Known'}}
  { };

  template<class T1, typename T2> struct foo5 :
    UnknownBase<T1,T2,ABC<T2,T1>> // expected-error {{unknown template name 'UnknownBase'}} \
                                  // expected-error {{use '> >'}}
  { };

  template<class T1, typename T2> struct foo6 :
    UnknownBase<T1,ABC<T2,T1>>, // expected-error {{unknown template name 'UnknownBase'}} \
                                // expected-error {{use '> >'}}
    Known<T1>  // expected-error {{too few template arguments for class template 'Known'}}
  { };

  template<class T1, typename T2, int N> struct foo7 :
    UnknownBase<T1,T2,(N>1)> // expected-error {{unknown template name 'UnknownBase'}}
  { };

  template<class T1, typename T2> struct foo8 :
    UnknownBase<X<int,int>,X<int,int>> // expected-error {{unknown template name 'UnknownBase'}} \
                                       // expected-error {{use '> >'}}
  { };

  template<class T1, typename T2> struct foo9 :
    UnknownBase<Known<int,int>,X<int,int>> // expected-error {{unknown template name 'UnknownBase'}} \
                                           // expected-error {{use '> >'}}
  { };

  template<class T1, typename T2> struct foo10 :
    UnknownBase<Known<int,int>,X<int,X<int,int>>> // expected-error {{unknown template name 'UnknownBase'}} \
                                                  // expected-error {{use '> >'}}
  { };

  template<int N1, int N2> struct foo11 :
    UnknownBase<2<N1,N2<4> // expected-error {{unknown template name 'UnknownBase'}}
  { };

}

namespace PR18793 {
  template<typename T, T> struct S {};
  template<typename T> int g(S<T, (T())> *);
}
