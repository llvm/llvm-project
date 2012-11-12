// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace rdar8436162 {
  class ClsA {
  public:
    static void f();
    void g();
  };

  class ClsB : virtual private ClsA {
  public:
    using ClsA::f;
    using ClsA::g; // expected-note{{member found by ambiguous name lookup}}
  };

  class ClsF : virtual private ClsA {
  public:
    using ClsA::f;
    using ClsA::g; // expected-note{{member found by ambiguous name lookup}}
  };

  class ClsE : public ClsB, public ClsF {
    void test() { 
      f();
      g(); // expected-error{{member 'g' found in multiple base classes of different types}}
    }
  };
}
