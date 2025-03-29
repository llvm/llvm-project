// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace N { 
  enum { C };
  template<class T> class B {
    void f(T);
  }; 
}

template<class C> void N::B<C>::f(C) {
  C b;
}

namespace N {
  enum { D };
  namespace M {
    enum { C , D };
    template<typename C> class X {
      template<typename U> void f(C, U);

      template<typename D> void g(C, D) {
        C c;
        D d;
      }
    };

    struct Y {
      template<typename U> void f(U);      
    };
  }

  struct Y {
    template<typename D> void f(D);
  };
}

template<typename C> 
template<typename D>
void N::M::X<C>::f(C, D) {
  C c;
  D d;
}

template<typename C>
void N::M::Y::f(C) {
  C c;
}

template<typename D> 
void N::Y::f(D) {
  D d;
}

// Ensure we properly interleave the searches within classes and template parameter lists.
namespace SearchClassBetweenTemplateParameterLists {
  int AA, BB; // none of the below lookups should ever consider these

  struct Base {
    using AA = void;
    using BB = void;
  };
  struct BaseT : Base {
    using T = void;
  };
  struct BaseU : Base {
    using U = void;
  };

  template<typename T> struct A {
    using AA = void;
    template<typename U> struct B { // #defined-here
      using BB = void;
      void f(U);
      void g(U);
      void h(T);
      void i(T);
      template<typename V> void j(V);
      template<typename V> void k(U);

      // OK: these find the template parameter not the member.
      template<typename AA> void l(AA x) { AA aa; }
      template<typename BB> void m(BB x) { BB bb; }

      struct C : Base {
        // All OK; these find the template parameters.
        template<typename> void f(T x) { T t; }
        template<typename> void g(U x) { U u; }
        template<typename AA> void h(AA x) { AA aa; }
        template<typename BB> void i(BB x) { BB bb; }
      };

      struct CT : BaseT {
        template<typename> void f(T x) { // expected-error {{void}}
          T t; // expected-error {{incomplete}}
        }
        template<typename> void g(U x) { U u; }
        template<typename AA> void h(AA x) { AA aa; }
        template<typename BB> void i(BB x) { BB bb; }
      };

      struct CU : BaseU {
        template<typename> void f(T x) { T t; }
        template<typename> void g(U x) { // expected-error {{void}}
          U u; // expected-error {{incomplete}}
        }
        template<typename AA> void h(AA x) { AA aa; }
        template<typename BB> void i(BB x) { BB bb; }
      };
    };
  };

  // Search order for the below is:
  // 1) template parameter scope of the function itself (if any)
  // 2) class of which function is a member
  // 3) template parameter scope of inner class
  // 4) class of which class is a member
  // 5) template parameter scope of outer class

  // OK, 'AA' found in (3)
  template<typename T> template<typename AA>
  void A<T>::B<AA>::f(AA) {
    AA aa;
  }

  // error, 'BB' found in (2)
  template<typename T> template<typename BB>
  void A<T>::B<BB>::g(BB) { // expected-error {{does not match}}
    BB bb; // expected-error {{incomplete type}}
  } // expected-note@#defined-here {{defined here}}

  // error, 'AA' found in (4)
  template<typename AA> template<typename U>
  void A<AA>::B<U>::h(AA) { // expected-error {{does not match}}
    AA aa; // expected-error {{incomplete type}}
  } // expected-note@#defined-here {{defined here}}

  // error, 'BB' found in (2)
  template<typename BB> template<typename U>
  void A<BB>::B<U>::i(BB) { // expected-error {{does not match}}
    BB bb; // expected-error {{incomplete type}}
  } // expected-note@#defined-here {{defined here}}

  // OK, 'BB' found in (1)
  template<typename T> template<typename U> template<typename BB>
  void A<T>::B<U>::j(BB) {
    BB bb;
  }

  // error, 'BB' found in (2)
  template<typename T> template<typename BB> template<typename V>
  void A<T>::B<BB>::k(V) { // expected-error {{does not match}}
    BB bb; // expected-error {{incomplete type}}
  } // expected-note@#defined-here {{defined here}}

  int CC;
  template <typename> struct C;
  template <template<typename> typename> struct D;
#if __cplusplus >= 202002L
  template <bool CC> requires (CC) struct E;
  template <typename> struct F;
  template <typename> concept True = true;
#endif
}

template <typename CC>
struct SearchClassBetweenTemplateParameterLists::C {
  void foo(CC); // This should find the template type parameter.
};

template <template<typename> typename CC>
struct SearchClassBetweenTemplateParameterLists::D {
  template <typename AA>
  CC<AA> foo(CC<AA>);
};

#if __cplusplus >= 202002L

template <bool CC> requires (CC)
struct SearchClassBetweenTemplateParameterLists::E {
  void foo() requires (CC);
};

template <SearchClassBetweenTemplateParameterLists::True CC>
struct SearchClassBetweenTemplateParameterLists::F<CC> {
  void foo(CC);
};

#endif
