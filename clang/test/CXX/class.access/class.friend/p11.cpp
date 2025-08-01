// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  void foo() {
    void bar();
    class A {
      friend void bar();
    };
  }
}

namespace test1 {
  void foo() {
    class A {
      friend void bar(); // expected-error {{cannot define friend function in a local class definition}}
    };
  }
}

namespace test2 {
  void bar(); // expected-note 3{{'::test2::bar' declared here}}

  void foo() { // expected-note 2{{'::test2::foo' declared here}}
    struct S1 {
      friend void foo(); // expected-error {{cannot define friend function 'foo' in a local class definition; did you mean '::test2::foo'?}}
    };

    void foo(); // expected-note {{local declaration nearly matches}}
    struct S2 {
      friend void foo();
    };

    {
      struct S2 {
        friend void foo(); // expected-error {{cannot define friend function in a local class definition}}
      };
    }

    {
      int foo;
      struct S3 {
        friend void foo(); // expected-error {{cannot define friend function 'foo' in a local class definition; did you mean '::test2::foo'?}}
      };
    }

    struct S4 {
      friend void bar(); // expected-error {{cannot define friend function 'bar' in a local class definition; did you mean '::test2::bar'?}}
    };

    { void bar(); }
    struct S5 {
      friend void bar(); // expected-error {{cannot define friend function 'bar' in a local class definition; did you mean '::test2::bar'?}}
    };

    {
      void bar();
      struct S6 {
        friend void bar();
      };
    }

    struct S7 {
      void bar() { Inner::f(); }
      struct Inner {
        friend void bar();
        static void f() {}
      };
    };

    void bar(); // expected-note {{'bar' declared here}}
    struct S8 {
      struct Inner {
        friend void bar();
      };
    };

    struct S9 {
      struct Inner {
        friend void baz(); // expected-error {{cannot define friend function 'baz' in a local class definition; did you mean 'bar'?}}
      };
    };

    struct S10 {
      void quux() {}
      void foo() {
        struct Inner1 {
          friend void bar(); // expected-error {{cannot define friend function 'bar' in a local class definition; did you mean '::test2::bar'?}}
          friend void quux(); // expected-error {{cannot define friend function in a local class definition}}
        };

        void bar();
        struct Inner2 {
          friend void bar();
        };
      }
    };
  }
}
