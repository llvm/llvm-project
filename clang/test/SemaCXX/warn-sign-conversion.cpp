// RUN: %clang_cc1 -triple %itanium_abi_triple -verify -fsyntax-only -Wsign-conversion %s

// NOTE: When a 'enumeral mismatch' warning is implemented then expect several
// of the following cases to be impacted.

// namespace for anonymous enums tests
namespace test1 {
  enum { A };
  enum { B = -1 };

  template <typename T> struct Foo {
    enum { C };
    enum { D = ~0U };
  };

  enum { E = ~0U };

  void doit_anonymous( int i ) {
    int a1 = 1 ? i : A;
    int a2 = 1 ? A : i;

    int b1 = 1 ? i : B;
    int b2 = 1 ? B : i;

    int c1 = 1 ? i : Foo<bool>::C;
    int c2 = 1 ? Foo<bool>::C : i;

    int d1a = 1 ? i : Foo<bool>::D; // expected-warning {{test1::Foo<bool>::(anonymous enum at }}
    int d1b = 1 ? i : Foo<bool>::D; // expected-warning {{warn-sign-conversion.cpp:13:5)' to 'int'}}
    int d2a = 1 ? Foo<bool>::D : i; // expected-warning {{operand of ? changes signedness: 'test1::Foo<bool>::(anonymous enum at }}
    int d2b = 1 ? Foo<bool>::D : i; // expected-warning {{warn-sign-conversion.cpp:13:5)' to 'int'}}
    int d3a = 1 ? B : Foo<bool>::D; // expected-warning {{operand of ? changes signedness: 'test1::Foo<bool>::(anonymous enum at }}
    int d3b = 1 ? B : Foo<bool>::D; // expected-warning {{warn-sign-conversion.cpp:13:5)' to 'int'}}
    int d4a = 1 ? Foo<bool>::D : B; // expected-warning {{operand of ? changes signedness: 'test1::Foo<bool>::(anonymous enum at }}
    int d4b = 1 ? Foo<bool>::D : B; // expected-warning {{warn-sign-conversion.cpp:13:5)' to 'int'}}

    int e1a = 1 ? i : E; // expected-warning {{operand of ? changes signedness: 'test1::(anonymous enum at }}
    int e1b = 1 ? i : E; // expected-warning {{warn-sign-conversion.cpp:16:3)' to 'int'}}
    int e2a = 1 ? E : i; // expected-warning {{operand of ? changes signedness: 'test1::(anonymous enum at }}
    int e2b = 1 ? E : i; // expected-warning {{warn-sign-conversion.cpp:16:3)' to 'int'}}
    int e3a = 1 ? E : B; // expected-warning {{operand of ? changes signedness: 'test1::(anonymous enum at }}
    int e3b = 1 ? E : B; // expected-warning {{warn-sign-conversion.cpp:16:3)' to 'int'}}
    int e4a = 1 ? B : E; // expected-warning {{operand of ? changes signedness: 'test1::(anonymous enum at }}
    int e4b = 1 ? B : E; // expected-warning {{warn-sign-conversion.cpp:16:3)' to 'int'}}
  }
}

// namespace for named enums tests
namespace test2 {
  enum Named1 { A };
  enum Named2 { B = -1 };

  template <typename T> struct Foo {
    enum Named3 { C };
    enum Named4 { D = ~0U };
  };

  enum Named5 { E = ~0U };

  void doit_anonymous( int i ) {
    int a1 = 1 ? i : A;
    int a2 = 1 ? A : i;

    int b1 = 1 ? i : B;
    int b2 = 1 ? B : i;

    int c1 = 1 ? i : Foo<bool>::C;
    int c2 = 1 ? Foo<bool>::C : i;

    int d1 = 1 ? i : Foo<bool>::D; // expected-warning {{operand of ? changes signedness: 'test2::Foo<bool>::Named4' to 'int'}}
    int d2 = 1 ? Foo<bool>::D : i; // expected-warning {{operand of ? changes signedness: 'test2::Foo<bool>::Named4' to 'int'}}
    int d3 = 1 ? B : Foo<bool>::D; // expected-warning {{operand of ? changes signedness: 'test2::Foo<bool>::Named4' to 'int'}}
    int d4 = 1 ? Foo<bool>::D : B; // expected-warning {{operand of ? changes signedness: 'test2::Foo<bool>::Named4' to 'int'}}

    int e1 = 1 ? i : E; // expected-warning {{operand of ? changes signedness: 'test2::Named5' to 'int'}}
    int e2 = 1 ? E : i; // expected-warning {{operand of ? changes signedness: 'test2::Named5' to 'int'}}
    int e3 = 1 ? E : B; // expected-warning {{operand of ? changes signedness: 'test2::Named5' to 'int'}}
    int e4 = 1 ? B : E; // expected-warning {{operand of ? changes signedness: 'test2::Named5' to 'int'}}
  }
}
