// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct X0 : T::template apply<U> { 
  X0(U u) : T::template apply<U>(u) { }
};

template<typename T, typename U>
struct X1 : T::apply<U> { }; // expected-error{{use 'template' keyword to treat 'apply' as a dependent template name}}

template<typename T>
struct X2 : vector<T> { }; // expected-error{{no template named 'vector'}}

namespace PR6031 {
  template<typename T>
  struct A;

  template <class X>
  struct C { };

  template <class TT>
  struct II {
    typedef typename A<TT>::type type;
  };

  template <class TT>
  struct FI : II<TT>
  {
    C<typename FI::type> a;
  };

  template <class TT>
  struct FI2
  {
    C<typename FI2::type> a; // expected-error{{no type named 'type' in 'FI2<TT>'}}
  };

  template<typename T>
  struct Base {
    class Nested { };
    template<typename U> struct MemberTemplate { };
    int a;
  };

  template<typename T>
  struct HasDepBase : Base<T> {
    int foo() {
      class HasDepBase::Nested nested;
      typedef typename HasDepBase::template MemberTemplate<T>::type type;
      return HasDepBase::a;
    }
  };

  template<typename T>
  struct NoDepBase {
    int foo() {
      class NoDepBase::Nested nested; // expected-error{{no class named 'Nested' in 'NoDepBase<T>'}}
      typedef typename NoDepBase::template MemberTemplate<T>::type type; // expected-error{{no member named 'MemberTemplate' in 'NoDepBase<T>'}}
      return NoDepBase::a; // expected-error{{no member named 'a' in 'NoDepBase<T>'}}
    }
  };
}

namespace Ambig {
  template<typename T>
  struct Base1 {
    typedef int type; // expected-note{{member type 'int' found by ambiguous name lookup}}
  };

  struct Base2 {
    typedef float type; // expected-note{{member type 'float' found by ambiguous name lookup}}
  };

  template<typename T>
  struct Derived : Base1<T>, Base2 {
    typedef typename Derived::type type; // expected-error{{member 'type' found in multiple base classes of different types}}
    type *foo(float *fp) { return fp; }
  };

  Derived<int> di; // expected-note{{instantiation of}}
}

namespace PR6081 {
  template<typename T>
  struct A { };

  template<typename T>
  class B : public A<T> 
  {
  public:
    template< class X >
    void f0(const X & k)
    {
      this->template f1<int>()(k);
    }
  };

  template<typename T>
  class C
  {
  public:
    template< class X >
    void f0(const X & k)
    {
      this->template f1<int>()(k); // expected-error{{no member named 'f1' in 'C<T>'}}
    }
  };
}

namespace PR6413 {
  template <typename T> class Base_A { };
  
  class Base_B { };
  
  template <typename T>
  class Derived
    : public virtual Base_A<T>
    , public virtual Base_B
  { };
}

namespace PR5812 {
  template <class T> struct Base { 
    Base* p; 
  }; 

  template <class T> struct Derived: public Base<T> { 
    typename Derived::Base* p; // meaning Derived::Base<T> 
  };

  Derived<int> di;
}
