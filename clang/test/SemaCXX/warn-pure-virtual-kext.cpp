// RUN: %clang_cc1 %s -fapple-kext -fsyntax-only -verify

struct A {
    virtual void f() = 0; // expected-note {{'f' declared here}}
    A() {
        A::f(); // expected-warning {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'A'}} // expected-note {{qualified call to 'A'::'f' is treated as a virtual call to 'f' due to -fapple-kext}}
    }
};

template <typename T> struct TA {
  virtual void f() = 0; // expected-note {{'f' declared here}}

  TA() { TA::f(); } // expected-warning {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'TA<int>'}} // expected-note {{qualified call to 'TA<int>'::'f' is treated as a virtual call to 'f' due to -fapple-kext}}
};

struct B : TA<int> { // expected-note {{in instantiation of member function 'TA<int>::TA' requested here}}
  void f() override;
};

B b;
