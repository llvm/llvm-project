// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s


// This test creates cases where implicit instantiations of various entities
// would cause a diagnostic, but provides expliict specializations for those
// entities that avoid the diagnostic. The specializations are alternately
// declarations and definitions, and the intent of this test is to verify
// that we allow specializations only in the appropriate namespaces (and
// nowhere else).
struct NonDefaultConstructible {
  NonDefaultConstructible(int);
};


// C++ [temp.expl.spec]p1:
//   An explicit specialization of any of the following:

//     -- function template
namespace N0 {
  template<typename T> void f0(T) {
    T t;
  }

  template<> void f0(NonDefaultConstructible) { }

  void test_f0(NonDefaultConstructible NDC) {
    f0(NDC);
  }
  
  template<> void f0(int);
  template<> void f0(long);
}

template<> void N0::f0(int) { } // okay

namespace N1 {
  template<> void N0::f0(long) { } // expected-error{{does not enclose namespace}}
}

template<> void N0::f0(double);

template<> void N0::f0(double) { }

struct X1 {
  template<typename T> void f(T);
  
  template<> void f(int);
};

//     -- class template
namespace N0 {
  
template<typename T>
struct X0 { // expected-note {{explicitly specialized declaration is here}}
  static T member;
  
  void f1(T t) {
    t = 17;
  }
  
  struct Inner : public T { }; // expected-note 2{{explicitly specialized declaration is here}}
  
  template<typename U>
  struct InnerTemplate : public T { }; // expected-note {{explicitly specialized declaration is here}}
  // expected-error@-1 {{base specifier must name a class}}
  
  template<typename U>
  void ft1(T t, U u);
};

}

template<typename T> 
template<typename U>
void N0::X0<T>::ft1(T t, U u) {
  t = u;
}

template<typename T> T N0::X0<T>::member;

template<> struct N0::X0<void> { };
N0::X0<void> test_X0;

namespace N1 {
  template<> struct N0::X0<const void> { }; // expected-error{{not in a namespace enclosing 'N0'}}
}

namespace N0 {
  template<> struct X0<volatile void>;
}

template<> struct N0::X0<volatile void> { 
  void f1(void *);
};

//     -- member function of a class template
template<> void N0::X0<void*>::f1(void *) { }

void test_spec(N0::X0<void*> xvp, void *vp) {
  xvp.f1(vp);
}

namespace N0 {
  template<> void X0<volatile void>::f1(void *) { } // expected-error{{no function template matches}}

  template<> void X0<const volatile void*>::f1(const volatile void*);
}

void test_x0_cvvoid(N0::X0<const volatile void*> x0, const volatile void *cvp) {
  x0.f1(cvp); // okay: we've explicitly specialized
}

//     -- static data member of a class template
namespace N0 {
  // This actually tests p15; the following is a declaration, not a definition.
  template<> 
  NonDefaultConstructible X0<NonDefaultConstructible>::member;
  
  template<> long X0<long>::member = 17;

  template<> float X0<float>::member;
  
  template<> double X0<double>::member;
}

NonDefaultConstructible &get_static_member() {
  return N0::X0<NonDefaultConstructible>::member;
}

template<> int N0::X0<int>::member;

template<> float N0::X0<float>::member = 3.14f;

namespace N1 {
  template<> double N0::X0<double>::member = 3.14; // expected-error{{does not enclose namespace}}
}

//    -- member class of a class template
namespace N0 {
  
  template<>
  struct X0<void*>::Inner { };

  template<>
  struct X0<int>::Inner { };

  template<>
  struct X0<unsigned>::Inner;

  template<>
  struct X0<float>::Inner;

  template<>
  struct X0<double>::Inner; // expected-note{{forward declaration}}
}

template<>
struct N0::X0<long>::Inner { };

template<>
struct N0::X0<float>::Inner { };

namespace N1 {
  template<>
  struct N0::X0<unsigned>::Inner { }; // expected-error{{member class specialization}}

  template<>
  struct N0::X0<unsigned long>::Inner { }; // expected-error{{member class specialization}}
};

N0::X0<void*>::Inner inner0;
N0::X0<int>::Inner inner1;
N0::X0<long>::Inner inner2;
N0::X0<float>::Inner inner3;
N0::X0<double>::Inner inner4; // expected-error{{incomplete}}

//    -- member class template of a class template
namespace N0 {
  template<>
  template<>
  struct X0<void*>::InnerTemplate<int> { };
  
  template<> template<>
  struct X0<int>::InnerTemplate<int>; // expected-note{{forward declaration}}

  template<> template<>
  struct X0<int>::InnerTemplate<long>;

  template<> template<>
  struct X0<int>::InnerTemplate<double>;
}

template<> template<>
struct N0::X0<int>::InnerTemplate<long> { }; // okay

template<> template<>
struct N0::X0<int>::InnerTemplate<float> { };

namespace N1 {
  template<> template<>
  struct N0::X0<int>::InnerTemplate<double> { }; // expected-error{{enclosing}}
}

N0::X0<void*>::InnerTemplate<int> inner_template0;
N0::X0<int>::InnerTemplate<int> inner_template1; // expected-error{{incomplete}}
N0::X0<int>::InnerTemplate<long> inner_template2;
N0::X0<int>::InnerTemplate<unsigned long> inner_template3; // expected-note{{instantiation}}

//    -- member function template of a class template
namespace N0 {
  template<>
  template<>
  void X0<void*>::ft1(void*, const void*) { }
  
  template<> template<>
  void X0<void*>::ft1(void *, int);

  template<> template<>
  void X0<void*>::ft1(void *, unsigned);

  template<> template<>
  void X0<void*>::ft1(void *, long);
}

template<> template<>
void N0::X0<void*>::ft1(void *, unsigned) { } // okay

template<> template<>
void N0::X0<void*>::ft1(void *, float) { }

namespace N1 {
  template<> template<>
  void N0::X0<void*>::ft1(void *, long) { } // expected-error{{does not enclose namespace}}
}


void test_func_template(N0::X0<void *> xvp, void *vp, const void *cvp,
                        int i, unsigned u) {
  xvp.ft1(vp, cvp);
  xvp.ft1(vp, i);
  xvp.ft1(vp, u);
}

namespace PR8979 {
  template<typename Z>
  struct X0 {
    template <class T, class U> class Inner;
    struct OtherInner;
    template<typename T, typename U> void f(Inner<T, U>&);

    typedef Inner<OtherInner, OtherInner> MyInner;
    template<> void f(MyInner&);
  };
}
