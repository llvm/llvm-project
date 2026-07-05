// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() const; // expected-error {{non-member function cannot have 'const' qualifier}}
void (*pf)() const; // expected-error {{pointer to function type cannot have 'const' qualifier}}
extern void (&rf)() const; // expected-error {{reference to function type cannot have 'const' qualifier}}

typedef void cfn() const;
cfn f2; // expected-error {{non-member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

class C {
  void f() const;
  cfn f2;
  static void f3() const; // expected-error {{static member function cannot have 'const' qualifier}}
  static cfn f4; // expected-error {{static member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

  void m1() {
    x = 0;
  }

  void m2() const { // expected-note {{member function 'C::m2' is declared const here}}
    x = 0; // expected-error {{cannot assign to non-static data member within const member function 'm2'}}
  }

  int x;
};

void (C::*mpf)() const;
cfn C::*mpg;

// Don't crash!
void (PR14171)() const; // expected-error {{non-member function cannot have 'const' qualifier}}

// Test template instantiation of decayed array types.  Not really related to
// type quals.
template <typename T> void arrayDecay(const T a[]) { }
void instantiateArrayDecay() {
  int a[1];
  arrayDecay(a);
}

namespace GH79748 {
typedef decltype(sizeof(0)) size_t;
struct A {
  void* operator new(size_t bytes) const; //expected-error {{static member function cannot have 'const' qualifier}}
  void* operator new[](size_t bytes) const; //expected-error {{static member function cannot have 'const' qualifier}}

  void operator delete(void*) const; //expected-error {{static member function cannot have 'const' qualifier}}
  void operator delete[](void*) const; //expected-error {{static member function cannot have 'const' qualifier}}
};
struct B {
  void* operator new(size_t bytes) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
  void* operator new[](size_t bytes) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}

  void operator delete(void*) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
  void operator delete[](void*) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
};
}
