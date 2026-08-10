// RUN: %clang_cc1 -std=c++17 -verify -Wno-defaulted-function-deleted %s -triple x86_64-windows-msvc

// MSVC emits the complete destructor as if it were its own special member.
// Clang attempts to do the same. This affects the diagnostics clang emits,
// because deleting a type with a user declared constructor implicitly
// references the destructors of virtual bases, which might be deleted or access
// controlled.

namespace t1 {
struct A {
  ~A() = delete; // expected-note {{deleted here}}
};
struct B {
  B() = default;
  A o; // expected-note {{destructor of 'B' is implicitly deleted because field 'o' has a deleted destructor}}
};
struct C : virtual B {
  ~C(); // expected-error {{attempt to use a deleted function}}
};
void delete1(C *p) { delete p; } // expected-note {{in implicit destructor for 't1::C' first required here}}
void delete2(C *p) { delete p; }
}

namespace t2 {
struct A {
private:
  ~A();
};
struct B {
  B() = default;
  A o; // expected-note {{destructor of 'B' is implicitly deleted because field 'o' has an inaccessible destructor}}
};
struct C : virtual B {
  ~C(); // expected-error {{attempt to use a deleted function}}
};
void useCompleteDtor(C *p) { delete p; } // expected-note {{in implicit destructor for 't2::C' first required here}}
}

namespace t3 {
template <unsigned N>
class Base { ~Base(); }; // expected-note 1{{declared private here}}
// No diagnostic.
class Derived0 : virtual Base<0> { ~Derived0(); };
class Derived1 : virtual Base<1> {};
// Emitting complete dtor causes a diagnostic.
struct Derived2 : // expected-error {{inherited virtual base class 'Base<2>' has private destructor}}
                  virtual Base<2> {
  ~Derived2();
};
void useCompleteDtor(Derived2 *p) { delete p; } // expected-note {{in implicit destructor for 't3::Derived2' first required here}}
}
