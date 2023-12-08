// RUN: %clang_cc1 -std=c++1z %s -verify -Wno-unused

namespace InExpr {
  namespace A {
    void typo_first_a(); // expected-note {{found}}
    template<typename T> void typo_first_b(); // expected-note 2{{declared here}}
  }
  void testA() { A::typo_first_a<int>(); } // expected-error {{'typo_first_a' does not name a template but is followed by template arguments; did you mean 'typo_first_b'?}}

  namespace B {
    void typo_first_b(); // expected-note {{found}}
  }
  void testB() { B::typo_first_b<int>(); } // expected-error {{'typo_first_b' does not name a template but is followed by template arguments; did you mean 'A::typo_first_b'?}}

  struct Base {
    template<typename T> static void foo(); // expected-note 4{{declared here}}
    int n;
  };
  struct Derived : Base {
    void foo(); // expected-note {{found}}
  };
  // We probably don't want to suggest correcting to .Base::foo<int>
  void testMember() { Derived().foo<int>(); } // expected-error-re {{does not name a template but is followed by template arguments{{$}}}}

  struct Derived2 : Base {
    void goo(); // expected-note {{found}}
  };
  void testMember2() { Derived2().goo<int>(); } // expected-error {{member 'goo' of 'InExpr::Derived2' is not a template; did you mean 'foo'?}}

  void no_correction() {
    int foo; // expected-note 3{{found}}

    foo<int>(); // expected-error {{'foo' does not name a template but is followed by template arguments; did you mean 'Base::foo'?}}
    foo<>(); // expected-error {{'foo' does not name a template but is followed by template arguments; did you mean 'Base::foo'?}}
    foo<Base *>(); // expected-error {{'foo' does not name a template but is followed by template arguments; did you mean 'Base::foo'?}}

    // These are valid expressions.
    foo<foo; // expected-warning {{self-comparison}}
    foo<int()>(0);
    foo<int(), true>(false);
    foo<Base{}.n;
  }
}
