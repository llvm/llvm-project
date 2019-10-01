// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// C++11 [basic.lookup.argdep]p2
//
// [...] If an associated namespace is an inline namespace (10.3.1), its
// enclosing namespace is also included in the set. If an associated
// namespace directly contains inline namespaces, those inline namespaces
// are also included in the set.

namespace test1 {
  namespace L {
    namespace M {
      inline namespace N {
        inline namespace O {
          struct S {};
          void f1(S);
        }
        void f2(S);
      }
      void f3(S);
    }
    void f4(M::S); // expected-note {{declared here}}
  }

  void test() {
    L::M::S s;
    f1(s); // ok
    f2(s); // ok
    f3(s); // ok
    f4(s); // expected-error {{use of undeclared}}
  }
}

namespace test2 {
  namespace L {
    struct S {};
    inline namespace M {
      inline namespace N {
        inline namespace O {
          void f1(S);
        }
        void f2(S);
      }
      void f3(S);
    }
    void f4(S);
  }

  void test() {
    L::S s;
    f1(s); // ok
    f2(s); // ok
    f3(s); // ok
    f4(s); // ok
  }
}
