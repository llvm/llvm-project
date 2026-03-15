// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Inherit a valid non-default ctor.
namespace NonDefaultCtorValid {
  struct A {
    A(const int &x) {}
  };

  struct B : A {
    using A::A;
  };

  struct C {
    struct B b;
    C() : b(0) {}
  };

  void test() {
    B b(0);
    C c;
  }
}

// Inherit an invalid non-default ctor.
// The inherited ctor is invalid because it is unable to initialize s.
namespace NonDefaultCtorInvalid {
  struct S {
    S() = delete;
  };
  struct A {
    A(const int &x) {}
  };

  struct B : A {
    using A::A;
    S s;
  };

  struct C {
    struct B b;
    C() : b(0) {} // expected-error{{constructor inherited by 'B' from base class 'A' is implicitly deleted}}
                  // expected-note@-6{{constructor inherited by 'B' is implicitly deleted because field 's' has a deleted corresponding constructor}}
                  // expected-note@-15{{'S' has been explicitly marked deleted here}}
  };
}

// Inherit a valid default ctor.
namespace DefaultCtorValid {
  struct A {
    A() {}
  };

  struct B : A {
    using A::A;
  };

  struct C {
    struct B b;
    C() {}
  };

  void test() {
    B b;
    C c;
  }
}

// Inherit an invalid default ctor.
// The inherited ctor is invalid because it is unable to initialize s.
namespace DefaultCtorInvalid {
  struct S {
    S() = delete;
  };
  struct A {
    A() {}
  };

  struct B : A {
    using A::A;
    S s;
  };

  struct C {
    struct B b; // expected-note{{default constructed field 'b' declared here}}
    C() {} // expected-error{{call to implicitly-deleted default constructor of 'struct B'}}
           // expected-note@-6{{default constructor of 'B' is implicitly deleted because field 's' has a deleted default constructor}}
           // expected-note@-15{{'S' has been explicitly marked deleted here}}
  };
}
