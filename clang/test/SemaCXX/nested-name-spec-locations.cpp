// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note: the formatting in this test case is intentionally funny, with
// nested-name-specifiers stretched out vertically so that we can
// match up diagnostics per-line and still verify that we're getting
// good source-location information.

namespace outer {
  namespace inner {
    template<typename T>
    struct X0 {
    };
  }
}

template<typename T>
struct add_reference {
  typedef T& type;
};

namespace outer_alias = outer;

template<typename T>
struct UnresolvedUsingValueDeclTester {
  using outer::inner::X0<
          typename add_reference<T>::type 
    * // expected-error{{declared as a pointer to a reference of type}}
        >::value;
};

UnresolvedUsingValueDeclTester<int> UnresolvedUsingValueDeclCheck; // expected-note{{in instantiation of template class}}

template<typename T>
struct UnresolvedUsingTypenameDeclTester {
  using outer::inner::X0<
          typename add_reference<T>::type 
    * // expected-error{{declared as a pointer to a reference of type}}
        >::value;
};

UnresolvedUsingTypenameDeclTester<int> UnresolvedUsingTypenameDeclCheck; // expected-note{{in instantiation of template class}}


template<typename T, typename U>
struct PseudoDestructorExprTester {
  void f(T *t) {
    t->T::template Inner<typename add_reference<U>::type 
      * // expected-error{{as a pointer to a reference of type}}
      >::Blarg::~Blarg();
  }
};

struct HasInnerTemplate {
  template<typename T>
  struct Inner;

  typedef HasInnerTemplate T;
};

void PseudoDestructorExprCheck(
                    PseudoDestructorExprTester<HasInnerTemplate, float> tester) {
  tester.f(0); // expected-note{{in instantiation of member function}}
}

template<typename T>
struct DependentScopedDeclRefExpr {
  void f() {
    outer_alias::inner::X0<typename add_reference<T>::type 
      * // expected-error{{as a pointer to a reference of type}}
      >::value = 17;
  }
};

void DependentScopedDeclRefExprCheck(DependentScopedDeclRefExpr<int> t) {
  t.f(); // expected-note{{in instantiation of member function}}
}


template<typename T>
struct TypenameTypeTester {
  typedef typename outer::inner::X0<
          typename add_reference<T>::type 
    * // expected-error{{declared as a pointer to a reference of type}}
        >::type type;
};

TypenameTypeTester<int> TypenameTypeCheck; // expected-note{{in instantiation of template class}}

template<typename T, typename U>
struct DependentTemplateSpecializationTypeTester {
  typedef typename T::template apply<typename add_reference<U>::type 
                                     * // expected-error{{declared as a pointer to a reference of type}}
                                     >::type type;
};

struct HasApply {
  template<typename T>
  struct apply {
    typedef T type;
  };
};

DependentTemplateSpecializationTypeTester<HasApply, int> DTSTCheck; // expected-note{{in instantiation of template class}}

template<typename T, typename U>
struct DependentTemplateSpecializationTypeTester2 {
  typedef typename T::template apply<typename add_reference<U>::type 
                                     * // expected-error{{declared as a pointer to a reference of type}}
                                     > type;
};

DependentTemplateSpecializationTypeTester2<HasApply, int> DTSTCheck2; // expected-note{{in instantiation of template class}}

template<typename T, typename U>
struct DependentTemplateSpecializationTypeTester3 :
  T::template apply<typename add_reference<U>::type 
                                     * // expected-error{{declared as a pointer to a reference of type}}
                                     >
{};

DependentTemplateSpecializationTypeTester3<HasApply, int> DTSTCheck3; // expected-note{{in instantiation of template class}}

template<typename T, typename U>
struct DependentTemplateSpecializationTypeTester4 {
  typedef class T::template apply<typename add_reference<U>::type 
                                     * // expected-error{{declared as a pointer to a reference of type}}
                                     > type;
};

DependentTemplateSpecializationTypeTester4<HasApply, int> DTSTCheck4; // expected-note{{in instantiation of template class}}

template<template<class T> class TTP>
struct AcceptedTemplateTemplateParameter {
};

template<typename T, typename U>
struct DependentTemplateTemplateArgumentTester {
  typedef AcceptedTemplateTemplateParameter<
            T::
            template apply<
              typename add_reference<U>::type
              * // expected-error{{declared as a pointer to a reference of type}}
            >::
            template X>
    type;
};

DependentTemplateTemplateArgumentTester<HasApply, int> DTTACheck; // expected-note{{in instantiation of template class}}

namespace PR9388 {
  namespace std {
    template<typename T>     class vector     {
    };
  }
  template<typename T> static void foo(std::vector<T*> &V) {
    __PRETTY_FUNCTION__; // expected-warning{{expression result unused}}
  }
  void bar(std::vector<int*> &Blocks) {
    foo(Blocks); // expected-note{{in instantiation of}}
  }

}
