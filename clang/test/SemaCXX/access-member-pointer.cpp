// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR7694

class A { };
class B : private A { public: void foo(); }; // expected-note {{declared private here}}
void B::foo() {
  (void)static_cast<void(A::*)()>(&B::foo);
}
void bar() {
  (void)static_cast<void(A::*)()>(&B::foo); // expected-error {{cannot cast 'B' to its private base class 'A'}}
}
