// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s

namespace basic {
struct C {
  static void foo2() {}
};
template <typename T>
struct A {
  typedef C D;
};

template <typename T>
struct B : A<T> {
  void foo() {
    D::foo2(); // expected-warning {{use of undeclared identifier 'D'; unqualified lookup into dependent bases of class template 'B' is a Microsoft extension}}
  }
};

template struct B<int>; // Instantiation has no warnings.
}

namespace nested_nodep_base {
// There are limits to our hacks, MSVC accepts this, but we don't.
struct A {
  struct D { static void foo2(); };
};
template <typename T>
struct B : T {
  struct C {
    void foo() {
      D::foo2(); // expected-error {{use of undeclared identifier 'D'}}
    }
  };
};

template struct B<A>; // Instantiation has no warnings.
}

namespace nested_dep_base {
// We actually accept this because the inner class has a dependent base even
// though it isn't a template.
struct A {
  struct D { static void foo2(); };
};
template <typename T>
struct B {
  struct C : T {
    void foo() {
      D::foo2(); // expected-warning {{use of undeclared identifier 'D'; unqualified lookup into dependent bases of class template 'C' is a Microsoft extension}}
    }
  };
};

template struct B<A>; // Instantiation has no warnings.
}
