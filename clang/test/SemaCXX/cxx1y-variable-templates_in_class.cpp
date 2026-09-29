// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++98 -verify -fsyntax-only %s -Wno-c++11-extensions -Wno-c++1y-extensions -DPRECXX11
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -verify -fsyntax-only -Wno-c++1y-extensions %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++1y -verify -fsyntax-only %s -DCPP1Y

#define CONST const

#ifdef PRECXX11
#define static_assert _Static_assert
#endif

class A {
  template<typename T> CONST T wrong;           // expected-error {{non-static data member 'wrong' cannot be declared as a template}}
  template<typename T> CONST T wrong_init = 5;      // expected-error {{non-static data member 'wrong_init' cannot be declared as a template}}
  template<typename T, typename T0> static CONST T right = T(100);
  template<typename T> static CONST T right<T,int> = 5;
  template<typename T> CONST int right<int,T>;  // expected-error {{non-static data member 'right' cannot be declared as a template}}
  template<typename T> CONST float right<float,T> = 5;  // expected-error {{non-static data member 'right' cannot be declared as a template}}
#ifdef PRECXX11
                                                        // expected-warning@-2 {{in-class initializer for static data member of type 'const float' is a GNU extension}}
#else
                                                        // expected-error@-4 {{in-class initializer for static data member of type 'const float' requires 'constexpr' specifier}}
                                                        // expected-note@-5 {{add 'constexpr'}}
#endif
  template<> CONST int right<int,int> = 7;
  template<> CONST float right<float,int>;
  template static CONST int right<int,int>;     // expected-error {{expected '<' after 'template'}}
};

namespace out_of_line {
  class B0 {
    template<typename T, typename T0> static CONST T right = T(100);
    template<typename T> static CONST T right<T,int> = T(5);
  };
  template<> CONST int B0::right<int,int> = 7; // expected-note {{previous}}
  template CONST int B0::right<int,int>; // expected-warning {{has no effect}}
  template<> CONST int B0::right<int,float>; // expected-note {{previous}}
  template CONST int B0::right<int,float>; // expected-warning {{has no effect}}

  class B1 {
    template<typename T, typename T0> static CONST T right;
    template<typename T> static CONST T right<T,int>;
  };
  template<typename T, typename T0> CONST T B1::right = T(100);
  template<typename T> CONST T B1::right<T,int> = T(5);

  class B2 {
    template<typename T, typename T0> static CONST T right = T(100);  // expected-note {{previous initialization is here}}
    template<typename T> static CONST T right<T,int> = T(5);          // expected-note {{previous initialization is here}}
  };
  template<typename T, typename T0> CONST T B2::right = T(100);   // expected-error {{static data member 'right' already has an initializer}}
  template<typename T> CONST T B2::right<T,int> = T(5);           // expected-error {{static data member 'right' already has an initializer}}

  class B3 {
    template<typename T, typename T0> static CONST T right = T(100);
    template<typename T> static CONST T right<T,int> = T(5);
  };
  template<typename T, typename T0> CONST T B3::right;
  template<typename T> CONST T B3::right<T,int>;

  class B4 {
    template<typename T, typename T0> static CONST T a;
    template<typename T> static CONST T a<T,int> = T(100);
    template<typename T, typename T0> static CONST T b = T(100);
    template<typename T> static CONST T b<T,int>;
  };
  template<typename T, typename T0> CONST T B4::a; // expected-error {{default initialization of an object of const type 'const int'}}
  template<typename T> CONST T B4::a<T,int>;
  template CONST int B4::a<int,char>; // expected-note {{in instantiation of}}
  template CONST int B4::a<int,int>;

  template<typename T, typename T0> CONST T B4::b;
  template<typename T> CONST T B4::b<T,int>; // expected-error {{default initialization of an object of const type 'const int'}}
  template CONST int B4::b<int,char>;
  template CONST int B4::b<int,int>; // expected-note {{in instantiation of}}
}

namespace non_const_init {
  class A {
    template<typename T> static T wrong_inst_undefined = T(10); // expected-note {{refers here}}
    template<typename T> static T wrong_inst_defined = T(10); // expected-error {{non-const static data member must be initialized out of line}}
    template<typename T> static T wrong_inst_out_of_line;
  };

  template const int A::wrong_inst_undefined<const int>; // expected-error {{undefined}}

  template<typename T> T A::wrong_inst_defined;
  template const int A::wrong_inst_defined<const int>;
  template int A::wrong_inst_defined<int>; // expected-note {{in instantiation of static data member 'non_const_init::A::wrong_inst_defined<int>' requested here}}

  template<typename T> T A::wrong_inst_out_of_line = T(10);
  template int A::wrong_inst_out_of_line<int>;

  class B {
    template<typename T> static T wrong_inst; // expected-note {{refers here}}
    template<typename T> static T wrong_inst<T*> = T(100); // expected-error {{non-const static data member must be initialized out of line}} expected-note {{refers here}}

    template<typename T> static T wrong_inst_fixed;
    template<typename T> static T wrong_inst_fixed<T*>;
  };
  template int B::wrong_inst<int>; // expected-error {{undefined}}
  // FIXME: It'd be better to produce the 'explicit instantiation of undefined
  // template' diagnostic here, not the 'must be initialized out of line'
  // diagnostic.
  template int B::wrong_inst<int*>; // expected-note {{in instantiation of static data member 'non_const_init::B::wrong_inst<int *>' requested here}}
  template const int B::wrong_inst<const int*>; // expected-error {{undefined}}
  template<typename T> T B::wrong_inst_fixed = T(100);
  template int B::wrong_inst_fixed<int>;

  class C {
    template<typename T> static CONST T right_inst = T(10); // expected-note {{here}}
    template<typename T> static CONST T right_inst<T*> = T(100); // expected-note {{here}}
  };
  template CONST int C::right_inst<int>; // expected-error {{undefined variable template}}
  template CONST int C::right_inst<int*>; // expected-error {{undefined variable template}}

  namespace pointers {

    struct C0 {
      template<typename U> static U Data;
      template<typename U> static CONST U Data<U*> = U(); // expected-note {{here}}

      template<typename U> static U Data2;
      template<typename U> static CONST U Data2<U*> = U();
    };
    const int c0_test = C0::Data<int*>;
    static_assert(c0_test == 0, "");
    template const int C0::Data<int*>; // expected-error {{undefined}}

    template<typename U> const U C0::Data2<U*>;
    template const int C0::Data2<int*>;

    struct C1a {
      template<typename U> static U Data;
      template<typename U> static U* Data<U*>;   // Okay, with out-of-line definition
    };
    template<typename T> T* C1a::Data<T*> = new T();
    template int* C1a::Data<int*>;

    struct C1b {
      template<typename U> static U Data;
      template<typename U> static CONST U* Data<U*>;   // Okay, with out-of-line definition
    };
    template<typename T> CONST T* C1b::Data<T*> = (T*)(0);
    template CONST int* C1b::Data<int*>;

    struct C2a {
      template<typename U> static int Data;
      template<typename U> static U* Data<U*> = new U();   // expected-error {{non-const static data member must be initialized out of line}}
    };
    template int* C2a::Data<int*>; // expected-note {{in instantiation of static data member 'non_const_init::pointers::C2a::Data<int *>' requested here}}

    struct C2b {
      template<typename U> static int Data;
      template<typename U> static U *const Data<U*> = (U*)(0); // expected-error {{static data member of type 'int *const'}}
    };
    template<typename U> U *const C2b::Data<U*>;
    template int *const C2b::Data<int*>; // expected-note {{in instantiation of static data member 'non_const_init::pointers::C2b::Data<int *>' requested here}}
  }
}

#ifndef PRECXX11
namespace constexpred {
  class A {
    template<typename T> constexpr T wrong;           // expected-error {{non-static data member 'wrong' cannot be declared as a template}}
                                                      // expected-error@-1 {{declaration of constexpr static data member 'wrong' requires an initializer}}
    template<typename T> constexpr T wrong_init = 5;  // expected-error {{non-static data member 'wrong_init' cannot be declared as a template}}
    template<typename T, typename T0> static constexpr T right = T(100);
    template<typename T> static constexpr T right<T,int> = 5;
    template<typename T> constexpr int right<int,T>;         // expected-error {{non-static data member 'right' cannot be declared as a template}}
                                                             // expected-error@-1 {{declaration of constexpr static data member 'right<int, T>' requires an initializer}}
    template<typename T> constexpr float right<float,T> = 5; // expected-error {{non-static data member 'right' cannot be declared as a template}}
    template<> constexpr int right<int,int> = 7;
    template<> constexpr float right<float, int>; // expected-error {{declaration of constexpr static data member 'right<float, int>' requires an initializer}}
    template static constexpr int right<int,int>;     // expected-error {{expected '<' after 'template'}}
  };
}
#endif

namespace in_class_template {

  template<typename T>
  class D0 {
    template<typename U> static U Data; // expected-note {{here}}
    template<typename U> static CONST U Data<U*> = U();
  };
  template CONST int D0<float>::Data<int*>;
  template int D0<float>::Data<int>; // expected-error {{undefined}}
  template<typename T> template<typename U> const U D0<T>::Data<U*>;

  template<typename T>
  class D1 {
    template<typename U> static U Data;
    template<typename U> static U* Data<U*>;
  };
  template<typename T>
  template<typename U> U* D1<T>::Data<U*> = (U*)(0);
  template int* D1<float>::Data<int*>; // expected-note {{previous}}
  template int* D1<float>::Data<int*>; // expected-error {{duplicate explicit instantiation}}

  template<typename T>
  class D2 {
    template<typename U> static U Data;
    template<typename U> static U* Data<U*>;
  };
  template<>
  template<typename U> U* D2<float>::Data<U*> = (U*)(0) + 1;
  template int* D2<float>::Data<int*>; // expected-note {{previous}}
  template int* D2<float>::Data<int*>; // expected-error {{duplicate explicit instantiation}}

  template<typename T>
  struct D3 {
    template<typename U> static CONST U Data = U(100); // expected-note {{here}}
  };
  static_assert(D3<float>::Data<int> == 100, "");
  template const char D3<float>::Data<char>; // expected-error {{undefined}}

  namespace bug_files {
    template<typename T>
    class D0a {
      template<typename U> static U Data;
      template<typename U> static CONST U Data<U*> = U(10);  // expected-note {{previous declaration is here}}
    };
    template<>
    template<typename U> U D0a<float>::Data<U*> = U(100);  // expected-error {{redefinition of 'Data'}}

    // FIXME: We should accept this, and the corresponding case for class
    // templates.
    //
    // [temp.class.spec.mfunc]/2: If the primary member template is explicitly
    // specialized for a given specialization of the enclosing class template,
    // the partial specializations of the member template are ignored
    template<typename T>
    class D1 {
      template<typename U> static U Data;
      template<typename U> static CONST U Data<U*> = U(10);  // expected-note {{previous declaration is here}}
    };
    template<>
    template<typename U> U D1<float>::Data = U(10);
    template<>
    template<typename U> U D1<float>::Data<U*> = U(100);  // expected-error{{redefinition of 'Data'}}
  }

  namespace definition_after_outer_instantiation {
    template<typename A> struct S {
      template<typename B> static const int V1;
      template<typename B> static const int V2; // expected-note 3{{here}}
    };
    template struct S<int>;
    template<typename A> template<typename B> const int S<A>::V1 = 123;
    template<typename A> template<typename B> const int S<A>::V2<B*> = 456;

    static_assert(S<int>::V1<int> == 123, "");

    // FIXME: The first and third case below possibly should be accepted. We're
    // not picking up partial specializations added after the primary template
    // is instantiated. This is kind of implied by [temp.class.spec.mfunc]/2,
    // and matches our behavior for member class templates, but it's not clear
    // that this is intentional. See PR17294 and core-24030.
    static_assert(S<int>::V2<int*> == 456, ""); // FIXME expected-error {{}} expected-note {{initializer of 'V2<int *>' is unknown}}
    static_assert(S<int>::V2<int&> == 789, ""); // expected-error {{}} expected-note {{initializer of 'V2<int &>' is unknown}}

    template<typename A> template<typename B> const int S<A>::V2<B&> = 789;
    static_assert(S<int>::V2<int&> == 789, ""); // FIXME expected-error {{}} expected-note {{initializer of 'V2<int &>' is unknown}}

    // All is OK if the partial specialization is declared before the implicit
    // instantiation of the class template specialization.
    static_assert(S<char>::V1<int> == 123, "");
    static_assert(S<char>::V2<int*> == 456, "");
    static_assert(S<char>::V2<int&> == 789, "");
  }

  namespace incomplete_array {
    template<typename T> extern T var[];
    template<typename T> T var[] = { 1, 2, 3 };
    template<> char var<char>[] = "hello";
    template<typename T> char var<T*>[] = "pointer";

    static_assert(sizeof(var<int>) == 12, "");
    static_assert(sizeof(var<char>) == 6, "");
    static_assert(sizeof(var<void*>) == 8, "");

    template<typename...> struct tuple;

    template<typename T> struct A {
      template<typename U> static T x[];
      template<typename U> static T y[];

      template<typename...U> static T y<tuple<U...> >[];
    };

    int *use_before_definition = A<int>::x<char>;
    template<typename T> template<typename U> T A<T>::x[sizeof(U)];
    static_assert(sizeof(A<int>::x<char>) == 4, "");

    template<typename T> template<typename...U> T A<T>::y<tuple<U...> >[] = { U()... };
    static_assert(sizeof(A<int>::y<tuple<char, char, char> >) == 12, "");
  }

  namespace bad_reference {
    struct S {
      template<typename T> static int A; // expected-note 4{{here}}
    };

    template<typename T> void f() {
      typename T::template A<int> a; // expected-error {{template name refers to non-type template 'in_class_template::bad_reference::S::template A'}}
    }
    template<typename T> void g() {
      T::template A<int>::B = 0; // expected-error {{template name refers to non-type template 'in_class_template::bad_reference::S::template A'}}
    }
    template<typename T> void h() {
      class T::template A<int> c; // expected-error {{template name refers to non-type template 'in_class_template::bad_reference::S::template A'}}
    }

    template<typename T>
    struct X : T::template A<int> {}; // expected-error {{template name refers to non-type template 'in_class_template::bad_reference::S::template A'}}

    template void f<S>(); // expected-note {{in instantiation of}}
    template void g<S>(); // expected-note {{in instantiation of}}
    template void h<S>(); // expected-note {{in instantiation of}}
    template struct X<S>; // expected-note {{in instantiation of}}
  }
}

namespace in_nested_classes {
  // TODO:
}

namespace bitfield {
struct S {
  template <int I>
  static int f : I; // expected-error {{static member 'f' cannot be a bit-field}}
};
}

namespace b20896909 {
  // This used to crash.
  template<typename T> struct helper {};
  template<typename T> class A {
    template <typename> static helper<typename T::error> x;  // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
  };
  void test() {
    A<int> ai;  // expected-note {{in instantiation of}}
  }
}
namespace member_access_is_ok {
#ifdef CPP1Y
  namespace ns1 {
    struct A {
      template<class T, T N> constexpr static T Var = N;
    };
    static_assert(A{}.Var<int,5> == 5,"");
  } // end ns1
#endif // CPP1Y

namespace ns2 {
  template<class T> struct A {
    template<class U, T N, U M> static T&& Var;
  };
  template<class T> template<class U, T N, U M> T&& A<T>::Var = T(N + M);
  int *AV = &A<int>().Var<char, 5, 'A'>;

} //end ns2
} // end ns member_access_is_ok

#ifdef CPP1Y
namespace PR24473 {
struct Value
{
    template<class T>
    static constexpr T value = 0;
};

template<typename TValue>
struct Something
{
    void foo() {
        static_assert(TValue::template value<int> == 0, ""); // error
    }
};

int main() {
    Something<Value>{}.foo();
    return 0;
}

} // end ns PR24473
#endif // CPP1Y

namespace dependent_static_var_template {
  struct A {
    template<int = 0> static int n; // expected-note 2{{here}}
  };
  int &r = A::template n; // expected-error {{use of variable template 'A::template n' requires template arguments}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

  template<typename T>
  int &f() { return T::template n; } // expected-error {{use of variable template 'dependent_static_var_template::A::template n' requires template arguments}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}
  int &s = f<A>(); // expected-note {{instantiation of}}

  namespace B {
    template<int = 0> static int n; // expected-note {{here}}
  }
  int &t = B::template n; // expected-error {{use of variable template 'B::template n' requires template arguments}} expected-error {{a template argument list is expected after a name prefixed by the template keyword}}

  struct C {
    template <class T> static T G;
  };
  template<class T> T C::G = T(6);

  template <class T> T F() {
    C c;
    return c.G<T>;
  }

  int cf() { return F<int>(); }

#ifdef CPP1Y
  namespace GH55872 {
    struct s {
      template<typename T>
      static CONST auto f = [] { return T::template g<s>; };
      // expected-note@-1 {{in instantiation of static data member 'dependent_static_var_template::GH55872::t::g' requested here}}
      // expected-note@-2 {{while substituting into a lambda expression here}}
    };

    struct t {
      template<typename T>
      static CONST auto g = [] { return T::template f<t>; };
      // expected-error@-1 {{the type of variable template specialization 'f<dependent_static_var_template::GH55872::t>' declared with deduced type 'const auto' depends on itself}}
      // expected-note@-2 {{while substituting into a lambda expression here}}
    };

    void test() { s::f<t>()(); } // expected-note {{in instantiation of static data member 'dependent_static_var_template::GH55872::s::f' requested here}}
  }
#endif
}

#ifndef PRECXX11
namespace template_vars_in_template {
template <int> struct TakesInt {};

template <class T2>
struct S {
  template <class T1>
  static constexpr int v = 42;

  template <class T>
  void mf() {
    constexpr int val = v<T>;
  }

  void mf2() {
    constexpr int val = v<char>;
    TakesInt<val> ti;
    (void)ti.x; // expected-error{{no member named 'x' in 'template_vars_in_template::TakesInt<42>'}}
  }
};

void useit() {
  S<int> x;
  x.mf<double>();
  x.mf2(); // expected-note{{in instantiation of member function 'template_vars_in_template::S<int>::mf2' requested here}}
}
}
#endif
