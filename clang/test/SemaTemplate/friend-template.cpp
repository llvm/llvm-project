// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR5057
namespace test0 {
  namespace std {
    class X {
    public:
      template<typename T> friend struct Y;
    };
  }

  namespace std {
    template<typename T> struct Y {};
  }
}

namespace test1 {
  template<typename T> void f1(T) { } // expected-note{{here}}

  class X {
    template<typename T> friend void f0(T);
    template<typename T> friend void f1(T);
  };

  template<typename T> void f0(T) { }
  template<typename T> void f1(T) { } // expected-error{{redefinition}}
}

// PR4768
namespace test2 {
  template<typename T> struct X0 {
    template<typename U> friend struct X0;
  };
  
  template<typename T> struct X0<T*> {
    template<typename U> friend struct X0;
  };

  template<> struct X0<int> {
    template<typename U> friend struct X0;
  };

  template<typename T> struct X1 {
    template<typename U> friend void f2(U);
    template<typename U> friend void f3(U);
  };

  template<typename U> void f2(U);

  X1<int> x1i;
  X0<int*> x0ip;

  template<> void f2(int);

  // FIXME: Should this declaration of f3 be required for the specialization of
  // f3<int> (further below) to work? GCC and EDG don't require it, we do...
  template<typename U> void f3(U);

  template<> void f3(int);
}

// PR5332
namespace test3 {
  template <typename T> class Foo {
    template <typename U>
    friend class Foo;
  };

  Foo<int> foo;

  template<typename T, T Value> struct X2a;

  template<typename T, int Size> struct X2b;

  template<typename T>
  class X3 {
    template<typename U, U Value> friend struct X2a;

    // FIXME: the redeclaration note ends up here because redeclaration
    // lookup ends up finding the friend target from X3<int>.
    template<typename U, T Value> friend struct X2b; // expected-error {{template non-type parameter has a different type 'long' in template redeclaration}} \
      // expected-note {{previous non-type template parameter with type 'int' is here}}
  };

  X3<int> x3i; // okay

  X3<long> x3l; // expected-note {{in instantiation}}
}

// PR5716
namespace test4 {
  template<typename> struct A {
    template<typename T> friend void f(const A<T>&);
  };

  template<typename T> void f(const A<T>&) {
    int a[sizeof(T) ? -1 : -1]; // expected-error {{array with a negative size}}
  }

  void f() {
    f(A<int>()); // expected-note {{in instantiation of function template specialization}}
  }
}

namespace test5 {
  class outer {
    class foo;
    template <typename T> friend struct cache;
  };
  class outer::foo {
    template <typename T> friend struct cache;
  };
}

// PR6022
namespace PR6022 {
  template <class T1, class T2 , class T3  > class A;

  namespace inner {
    template<class T1, class T2, class T3, class T> 
    A<T1, T2, T3>& f0(A<T1, T2, T3>&, T);
  } 

  template<class T1, class T2, class T3>
  class A {
    template<class U1, class U2, class U3, class T>  
    friend A<U1, U2, U3>& inner::f0(A<U1, U2, U3>&, T);
  };
}

namespace FriendTemplateDefinition {
  template<unsigned > struct int_c { };

  template<typename T>
  struct X {
    template<unsigned N>
    friend void f(X, int_c<N>) {
      int value = N;
    };
  };

  void test_X(X<int> x, int_c<5> i5) {
    f(x, i5);
  }
}

namespace PR7013a {
  template<class > struct X0
  {
    typedef int type;
  };
  template<typename > struct X1
  {
  };
  template<typename , typename T> struct X2
  {
    typename T::type e;
  };
  namespace N
  {
    template <typename = int, typename = X1<int> > struct X3
    {
      template <typename T1, typename T2, typename B> friend void op(X2<T1, T2>& , B);
    };
    template <typename Ch, typename Tr, typename B> void op(X2<Ch, Tr>& , B)
    {
      X2<int, Tr> s;
    }
  }
  int n()
  {
    X2<int, X0<int> > ngs;
    N::X3<> b;
    op(ngs, b);
    return 0;
  }
}

namespace PR7013b {
  template<class > struct X0
  {
    typedef int type;
  };
  template<typename > struct X1
  {
  };
  template<typename , typename T> struct X2
  {
    typename T::type e;
  };
  namespace N
  {
    template <typename = X1<int> > struct X3
    {
      template <typename T1, typename T2, typename B> friend void op(X2<T1, T2>& , B);
    };
    template <typename Ch, typename Tr, typename B> void op(X2<Ch, Tr>& , B)
    {
      X2<int, Tr> s;
    }
  }
  int n()
  {
    X2<int, X0<int> > ngs;
    N::X3<> b;
    op(ngs, b);
    return 0;
  }

}

namespace PR8649 {
  template<typename T, typename U, unsigned N>
  struct X {
    template<unsigned M> friend class X<T, U, M>; // expected-error{{partial specialization cannot be declared as a friend}}
  };

  X<int, float, 7> x;
}

// Don't crash, and error on invalid friend type template.
namespace friend_type_template_no_tag {
  template <typename T> struct S {
    template <typename U> friend S<U>; // expected-error{{friend type templates must use an elaborated type}}
  };
  template struct S<int>;
}

namespace PR10660 {
  struct A {
    template <> friend class B; // expected-error{{extraneous 'template<>' in declaration of class 'B'}}
  };
}

namespace rdar11147355 {
  template <class T>
  struct A {
    template <class U> class B;
    template <class S> template <class U> friend class A<S>::B; // expected-warning {{dependent nested name specifier 'A<S>::' for friend template declaration is not supported; ignoring this friend declaration}}
  private:
    int n; // expected-note {{here}}
  };

  template <class S> template <class U> class A<S>::B {
  public:
    // FIXME: This should be permitted.
    int f(A<S*> a) { return a.n; } // expected-error {{private}}
  };

  A<double>::B<double>  ab;
  A<double*> a;
  int k = ab.f(a); // expected-note {{instantiation of}}
}

namespace RedeclUnrelated {
  struct S {
    int packaged_task;
    template<typename> class future {
      template<typename> friend class packaged_task;
    };
    future<void> share;
  };
}

namespace PR12557 {
  template <typename>
  struct Foo;

  template <typename Foo_>
  struct Bar {
    typedef Foo_  Foo; // expected-note {{previous}}

    template <typename> friend struct Foo; // expected-error {{redefinition of 'Foo' as different kind of symbol}}
  };

  Bar<int> b;
}

namespace PR12585 {
  struct A { };
  template<typename> struct B {
    template<typename> friend class A::does_not_exist; // \
     // expected-error {{friend declaration of 'does_not_exist' does not match any declaration in 'PR12585::A'}}
  };

  struct C {
    template<typename> struct D;
  };
  template<typename> class E {
    int n;
    template<typename> friend struct C::D;
  };
  template<typename T> struct C::D {
    int f() {
      return E<int>().n;
    }
  };
  int n = C::D<void*>().f();

  struct F {
    template<int> struct G;
  };
  template<typename T> struct H {
    // FIXME: As with cases above, the note here is on an unhelpful declaration,
    // and should point to the declaration of G within F.
    template<T> friend struct F::G; // \
      // expected-error {{different type 'char' in template redeclaration}} \
      // expected-note {{previous}}
  };
  H<int> h1; // ok
  H<char> h2; // expected-note {{instantiation}}
}

// Ensure that we can still instantiate a friend function template
// after the friend declaration is instantiated during the delayed
// parsing of a member function, but before the friend function has
// been parsed.
namespace rdar12350696 {
  template <class T> struct A {
    void foo() {
      A<int> a;
    }
    template <class U> friend void foo(const A<U> & a) {
      int array[sizeof(T) == sizeof(U) ? -1 : 1]; // expected-error {{negative size}}
    }
  };

  void test() {
    A<int> b;
    foo(b); // expected-note {{in instantiation}}
  }
}

namespace StackUseAfterScope {
template <typename T> class Bar {};
class Foo {
  // Make sure this doesn't crash.
  template <> friend class Bar<int>; // expected-error {{template specialization declaration cannot be a friend}}
  bool aux;
};
}
