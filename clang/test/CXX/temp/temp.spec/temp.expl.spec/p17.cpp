// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T1> 
class A {
  template<class T2> class B {
    void mf();
  };
};

template<> template<> class A<int>::B<double>; 
template<> template<> void A<char>::B<char>::mf();

template<> void A<char>::B<int>::mf(); // expected-error{{requires 'template<>'}}

namespace test1 {
  template <class> class A {
    static int foo;
    static int bar;
  };
  typedef A<int> AA;
  
  template <> int AA::foo = 0; 
  int AA::bar = 1; // expected-error {{template specialization requires 'template<>'}}
  int A<float>::bar = 2; // expected-error {{template specialization requires 'template<>'}}

  template <> class A<double> { 
  public:
    static int foo;
    static int bar;    
  };

  typedef A<double> AB;
  template <> int AB::foo = 0; // expected-error{{extraneous 'template<>'}}
  int AB::bar = 1;
}
