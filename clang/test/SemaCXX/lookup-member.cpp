// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  class String; // expected-note {{target of using declaration}}
};

using A::String; // expected-note {{using declaration}}
class String; // expected-error {{conflicts with target of using declaration}}

union value {
char *String;
};

namespace UnambiguousStaticMemberTemplate {
  // A static member template is not ambiguous if found in multiple base class
  // subobjects.
  struct A { template<typename T> static void f(T); static void g(); };
  struct B : A { using A::f; using A::g; };
  struct C : A { using A::f; using A::g; };
  struct D : B, C {};
  void f(D d) { d.f(0); d.g(); }
}

namespace UnambiguousReorderedMembers {
  // Static members are not ambiguous if we find them in a different order in
  // multiple base classes.
  struct A { static void f(); };
  struct B { static void f(int); };
  struct C : A, B { using A::f; using B::f; }; // expected-note {{found}}
  struct D : B, A { using B::f; using A::f; };
  struct E : C, D {};
  void f(E e) { e.f(0); }

  // But a different declaration set in different base classes does result in ambiguity.
  struct X : B, A { using B::f; using A::f; static void f(int, int); }; // expected-note {{found}}
  struct Y : C, X {};
  void g(Y y) { y.f(0); } // expected-error {{found in multiple base classes of different types}}
}
