// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T1> class A { 
  template<class T2> class B {
    template<class T3> void mf1(T3); 
    void mf2();
  };
}; 

template<> template<class X>
class A<long>::B { }; // #defined-here

template<> template<> template<class T>
  void A<int>::B<double>::mf1(T t) { } 

template<> template<> template<class T>
void A<long>::B<double>::mf1(T t) { } // expected-error{{does not match}}
                                      // expected-note@#defined-here{{defined here}}

// FIXME: This diagnostic could probably be better.
template<class Y> template<>
  void A<Y>::B<double>::mf2() { } // expected-error{{does not refer}}
