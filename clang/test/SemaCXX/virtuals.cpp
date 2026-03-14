// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -std=c++11 %s

class A {
  virtual void f();
  virtual void g() = 0; // expected-note{{unimplemented pure virtual method 'g' in 'A'}}

  void h() = 0; // expected-error {{'h' is not virtual and cannot be declared pure}}
  void i() = 1; // expected-error {{initializer on function does not look like a pure-specifier}}
  void j() = 0u; // expected-error {{initializer on function does not look like a pure-specifier}}


  void k();

public:
  A(int);
};

virtual void A::k() { } // expected-error{{'virtual' can only be specified inside the class definition}}

class B : public A {
  // Needs to recognize that overridden function is virtual.
  void g() = 0;

  // Needs to recognize that function does not override.
  void g(int) = 0; // expected-error{{'g' is not virtual and cannot be declared pure}}
};

// Needs to recognize invalid uses of abstract classes.
A fn(A) // expected-error{{parameter type 'A' is an abstract class}} \
        // expected-error{{return type 'A' is an abstract class}}
{
  A a; // expected-error{{variable type 'A' is an abstract class}}
  (void)static_cast<A>(0); // expected-error{{allocating an object of abstract class type 'A'}}
  try {
  } catch(A) { // expected-error{{variable type 'A' is an abstract class}}
  }
}

namespace rdar9670557 {
  typedef int func(int);
  func *a();
  struct X {
    virtual func f = 0;
    virtual func (g) = 0;
    func *h = 0;
  };
}

namespace pr8264 {
  struct Test {
    virtual virtual void func();  // expected-warning {{duplicate 'virtual' declaration specifier}}
  };
}

namespace issue63503 {
struct Base {
  virtual ~Base() = default;
};

struct Derived final : Base {
  virtual ~Derived() = defaul; // expected-error {{use of undeclared identifier 'defaul'}}
} do_not_crash;
}

namespace VirtualFriend {
  // DR (filed but no number yet): reject meaningless pure-specifier on a friend declaration.
  struct A { virtual int f(); };
  struct B { friend int A::f() = 0; }; // expected-error {{friend declaration cannot have a pure-specifier}}
  struct C {
    virtual int f();
    friend int C::f() = 0; // expected-error {{friend declaration cannot have a pure-specifier}}
  };
}
