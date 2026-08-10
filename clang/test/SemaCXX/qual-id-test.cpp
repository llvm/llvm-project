// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace A
{
    namespace B
    {
        struct base
#if __cplusplus <= 199711L
        // expected-note@-2 {{lookup in the object type 'A::sub' refers here}}
#endif
        {
            void x() {}
            void y() {}
        };
    }

    struct member
    {
        void foo();
    };

    struct middleman
    {
        member * operator->() { return 0; }
    };

    struct sub : B::base
    {
        void x() {}
        middleman operator->() { return middleman(); }
    };
}

struct bad
{
  int x();
};

namespace C
{
    void fun()
    {
        A::sub a;

        a.x();
    
        a.sub::x();
        a.base::x();

        a.B::base::x(); // expected-error{{use of undeclared identifier 'B'}}

        a.A::sub::x();
        a.A::B::base::x();

        a.bad::x(); // expected-error{{'bad::x' is not a member of class 'A::sub'}}

        a->foo();
        a->member::foo();
        a->A::member::foo();
    }

    void fun2()
    {
        A::sub *a;

        a->x();

        a->sub::x();
        a->base::x();

        a->B::base::x(); // expected-error{{use of undeclared identifier 'B'}}

        a->A::sub::x();
        a->A::B::base::x();

        a->bad::x(); // expected-error{{'bad::x' is not a member of class 'A::sub'}}

        (*a)->foo();
        (*a)->member::foo();
        (*a)->A::member::foo();
    }

    void fun3()
    {
        int i;
        i.foo(); // expected-error{{member reference base type 'int' is not a structure or union}}
    }

    void fun4a() {
      A::sub *a;
      
      typedef A::member base;
#if __cplusplus <= 199711L
      // expected-note@-2 {{lookup from the current scope refers here}}
#endif
      a->base::x();
#if __cplusplus <= 199711L
      // expected-error@-2 {{lookup of 'base' in member access expression is ambiguous}}
#endif
    }

    void fun4b() {
      A::sub *a;
      
      typedef A::B::base base;
      a->base::x();
    }
  
    template<typename T>
    void fun5()
    {
        T a;
        a.x();
        a->foo();

        a.A::sub::x();
        a.A::B::base::x();
        a->A::member::foo();

        a.bad::x(); // expected-error{{'bad::x' is not a member of class 'A::sub'}}
    }

  void test_fun5() {
    fun5<A::sub>(); // expected-note{{instantiation}}
  }
  
  template<typename T>
  void fun6() {
    T a;
    a.sub::x();
    a.base::x();
    a->member::foo();
    a.B::base::x(); // expected-error{{use of undeclared identifier 'B'}}
   }
  
  void test_fun6() {
    fun6<A::sub>(); // expected-note{{instantiation}}
  }
  
}

// PR4703
struct a {
  int a;
  static int sa;
};

a a;

int a::sa = a.a; // expected-error {{invalid use of non-static data member 'a'}}


namespace PR6645 {
  typedef int foo;
  namespace Inner {
    typedef int PR6645::foo; // expected-error{{typedef declarator cannot be qualified}} \
    // expected-error{{cannot define or redeclare 'foo' here because namespace 'Inner' does not enclose namespace 'PR6645'}}
  }
}
