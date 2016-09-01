// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5907 {
  template<typename T> struct identity { typedef T type; };
  struct A { A(); }; 
  identity<A>::type::A() { }

  struct B { void f(); };
  template<typename T> struct C { typedef B type; };
  
  void C<int>::type::f() { }
}

namespace PR9421 {
  namespace N { template<typename T> struct S { void f(); }; }
  typedef N::S<int> T;
  namespace N { template<> void T::f() {} }
}

namespace PR8277 {
  template< typename S >
  struct C
  {
    template< int >
    void F( void )
    {
    }
  };

  template< typename S >
  struct D
  {
    typedef C< int > A;
  };

  typedef D< int >::A A;

  template<>
  template<>
  void A::F< 0 >( void )
  {
  }
}

namespace PR8277b {
  template<typename S> struct C {
    void f();
  };
  template<typename S> struct D {
    typedef C<int> A;
  };
  template<> void D<int>::A::f() {
  }
}

namespace PR8708 {
  template<typename T> struct A { 
    template<typename U> struct B {
      // #2
      void f();     
    }; 
  };  

  // #A specialize the member template for 
  // implicit instantiation of A<int>,
  // leaving the member template "unspecialized"
  // (14.7.3/16). Specialization uses the syntax
  // for explicit specialization (14.7.3/14)
  template<> template<typename U> 
  struct A<int>::B {
    // #1
    void g();
  };  

  // #1 define its function g. There is an enclosing
  // class template, so we write template<> for each 
  // specialized template (14.7.3/15).
  template<> template<typename U>
  void A<int>::B<U>::g() { }

  // #2 define the unspecialized member template's
  // f
  template<typename T> template<typename U>
  void A<T>::B<U>::f() { }


  // specialize the member template again, now
  // specializing the member too. This specializes
  // #A
  template<> template<>
  struct A<int>::B<int> { 
    // #3
    void h();
  };

  // defines #3. There is no enclosing class template, so
  // we write no "template<>".
  void A<int>::B<int>::h() { }

  void test() { 
    // calls #1
    A<int>::B<float> a; a.g(); 

    // calls #2
    A<float>::B<int> b; b.f();

    // calls #3
    A<int>::B<int> c; c.h();
  }
}

namespace PR9482 {
  namespace N1 {
    template <typename T> struct S {
      void foo() {}
    };
  }

  namespace N2 {
    typedef N1::S<int> X;
  }

  namespace N1 {
    template<> void N2::X::foo() {}
  }
}

namespace PR9668 {
  namespace First
  {
    template<class T>
    class Bar
    {
    protected:

      static const bool static_bool;
    };
  }

  namespace Second
  {
    class Foo;
  }

  typedef First::Bar<Second::Foo> Special;

  namespace
  First
  {
    template<>
    const bool Special::static_bool(false);
  }
}

namespace PR9877 {
  template<int>
  struct X
  {
    struct Y;
  };

  template<> struct X<0>::Y { static const int Z = 1; };
  template<> struct X<1>::Y { static const int Z = 1; };

  const int X<0>::Y::Z;
  template<> const int X<1>::Y::Z;  // expected-error{{extraneous 'template<>' in declaration of variable 'Z'}}
}

namespace PR9913 {
  template<class,class=int>struct S;
  template<class X>struct S<X> {
    template<class T> class F;
  };

  template<class A>
  template<class B>
  class S<A>::F{};
}

namespace template_class_spec_perClassDecl_nested
{
  template <typename T1> struct A {
    template <typename T2> struct B {
      template <typename T3> struct C {
        static void foo();
      };
    };
  };

  template <> struct A<int> {
    template <typename T2> struct B {
      template <typename T3> struct C {
        static void foo();
      };
    };
  };

  template <> template <typename T3> struct A<int>::B<int>::C {
    static void foo();
  };

  template <> template <> struct A<int>::B<int>::C<int> {
    static void foo();
  };

  template <> template<> template <typename T2> struct A<bool>::B<bool>::C {
    static void foo();
  };
}


namespace spec_vs_expl_inst {

  // Test all permutations of Specialization,
  // explicit instantiation Declaration, and explicit instantiation defInition.

  namespace SDI {  // PR11558
    template <typename STRING_TYPE> class BasicStringPiece;
    template <> class BasicStringPiece<int> { };
    extern template class BasicStringPiece<int>;
    template class BasicStringPiece<int>;
  }

  namespace SID {
    template <typename STRING_TYPE> class BasicStringPiece;
    template <> class BasicStringPiece<int> { }; // expected-note {{previous template specialization is here}}
    template class BasicStringPiece<int>;  // expected-note {{explicit instantiation definition is here}} expected-warning {{has no effect}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation declaration (with 'extern') follows explicit instantiation definition (without 'extern')}}
  }

  namespace ISD {
    template <typename STRING_TYPE> class BasicStringPiece;  // expected-note {{template is declared here}}
    template class BasicStringPiece<int>;  // expected-error {{explicit instantiation of undefined template 'spec_vs_expl_inst::ISD::BasicStringPiece<int>'}}
    template <> class BasicStringPiece<int> { };
    extern template class BasicStringPiece<int>;
  }

  namespace IDS {
    template <typename STRING_TYPE> class BasicStringPiece;  // expected-note {{template is declared here}}
    template class BasicStringPiece<int>;  // expected-error {{explicit instantiation of undefined template 'spec_vs_expl_inst::IDS::BasicStringPiece<int>'}}  // expected-note {{explicit instantiation definition is here}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation declaration (with 'extern') follows explicit instantiation definition (without 'extern')}}
    template <> class BasicStringPiece<int> { };
  }

  namespace DIS {
    template <typename STRING_TYPE> class BasicStringPiece;  // expected-note {{template is declared here}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation of undefined template 'spec_vs_expl_inst::DIS::BasicStringPiece<int>'}}
    template class BasicStringPiece<int>;
    template <> class BasicStringPiece<int> { };
  }

  namespace DSI {
    template <typename STRING_TYPE> class BasicStringPiece;  // expected-note {{template is declared here}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation of undefined template 'spec_vs_expl_inst::DSI::BasicStringPiece<int>'}}
    template <> class BasicStringPiece<int> { }; // expected-note {{previous}}
    template class BasicStringPiece<int>; // expected-warning {{has no effect}}
  }

  // The same again, with a defined template class.

  namespace SDI_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    template <> class BasicStringPiece<int> { };
    extern template class BasicStringPiece<int>;
    template class BasicStringPiece<int>;
  }

  namespace SID_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    template <> class BasicStringPiece<int> { }; // expected-note {{previous}}
    template class BasicStringPiece<int>;  // expected-note {{explicit instantiation definition is here}} expected-warning {{has no effect}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation declaration (with 'extern') follows explicit instantiation definition (without 'extern')}}
  }

  namespace ISD_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    template class BasicStringPiece<int>;  // expected-note {{explicit instantiation first required here}}
    template <> class BasicStringPiece<int> { };  // expected-error {{explicit specialization of 'spec_vs_expl_inst::ISD_WithDefinedTemplate::BasicStringPiece<int>' after instantiation}}
    extern template class BasicStringPiece<int>;
  }

  namespace IDS_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    template class BasicStringPiece<int>;  // expected-note {{explicit instantiation definition is here}} expected-note {{previous definition is here}}
    extern template class BasicStringPiece<int>;  // expected-error {{explicit instantiation declaration (with 'extern') follows explicit instantiation definition (without 'extern')}}
    template <> class BasicStringPiece<int> { };  // expected-error {{redefinition of 'spec_vs_expl_inst::IDS_WithDefinedTemplate::BasicStringPiece<int>'}}
  }

  namespace DIS_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    extern template class BasicStringPiece<int>;  // expected-note {{explicit instantiation first required here}}
    template class BasicStringPiece<int>;
    template <> class BasicStringPiece<int> { };  // expected-error {{explicit specialization of 'spec_vs_expl_inst::DIS_WithDefinedTemplate::BasicStringPiece<int>' after instantiation}}
  }

  namespace DSI_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    extern template class BasicStringPiece<int>;  // expected-note {{explicit instantiation first required here}}
    template <> class BasicStringPiece<int> { };  // expected-error {{explicit specialization of 'spec_vs_expl_inst::DSI_WithDefinedTemplate::BasicStringPiece<int>' after instantiation}}
    template class BasicStringPiece<int>;
  }

  // And some more random tests.

  namespace SII_WithDefinedTemplate {
    template <typename STRING_TYPE> class BasicStringPiece {};
    template <> class BasicStringPiece<int> { }; // expected-note {{previous}}
    template class BasicStringPiece<int>;  // expected-note {{previous explicit instantiation is here}} expected-warning {{has no effect}}
    template class BasicStringPiece<int>;  // expected-error {{duplicate explicit instantiation of 'BasicStringPiece<int>'}}
  }

  namespace SIS {
    template <typename STRING_TYPE> class BasicStringPiece;
    template <> class BasicStringPiece<int> { };  // expected-note {{previous definition is here}} expected-note {{previous}}
    template class BasicStringPiece<int>; // expected-warning {{has no effect}}
    template <> class BasicStringPiece<int> { };  // expected-error {{redefinition of 'spec_vs_expl_inst::SIS::BasicStringPiece<int>'}}
  }

  namespace SDS {
    template <typename STRING_TYPE> class BasicStringPiece;
    template <> class BasicStringPiece<int> { };  // expected-note {{previous definition is here}}
    extern template class BasicStringPiece<int>;
    template <> class BasicStringPiece<int> { };  // expected-error {{redefinition of 'spec_vs_expl_inst::SDS::BasicStringPiece<int>'}}
  }

  namespace SDIS {
    template <typename STRING_TYPE> class BasicStringPiece;
    template <> class BasicStringPiece<int> { };  // expected-note {{previous definition is here}}
    extern template class BasicStringPiece<int>;
    template class BasicStringPiece<int>;
    template <> class BasicStringPiece<int> { };  // expected-error {{redefinition of 'spec_vs_expl_inst::SDIS::BasicStringPiece<int>'}}
  }

}
