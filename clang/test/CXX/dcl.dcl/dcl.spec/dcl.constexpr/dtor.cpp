// RUN: %clang_cc1 -std=c++2a -verify %s

// p3: if the function is a constructor or destructor, its class shall not have
// any virtual base classes;
namespace vbase {
  struct A {};
  struct B : virtual A { // expected-note {{virtual}}
    constexpr ~B() {} // expected-error {{constexpr member function not allowed in struct with virtual base class}}
  };
}

// p3: its function-body shall not enclose
//  -- a goto statement
//  -- an identifier label
//  -- a variable of non-literal type or of static or thread storage duration
namespace contents {
  struct A {
    constexpr ~A() {
      goto x; // expected-error {{statement not allowed in constexpr function}}
      x: ;
    }
  };
  struct B {
    constexpr ~B() {
      x: ; // expected-error {{statement not allowed in constexpr function}}
    }
  };
  struct Nonlit { Nonlit(); }; // expected-note {{not literal}}
  struct C {
    constexpr ~C() {
      Nonlit nl; // expected-error {{non-literal}}
    }
  };
  struct D {
    constexpr ~D() {
      static int a; // expected-error {{static variable}}
    }
  };
  struct E {
    constexpr ~E() {
      thread_local int e; // expected-error {{thread_local variable}}
    }
  };
  struct F {
    constexpr ~F() {
      extern int f;
    }
  };
}

// p5: for every subobject of class type or (possibly multi-dimensional) array
// thereof, that class type shall have a constexpr destructor
namespace subobject {
  struct A {
    ~A();
  };
  struct B : A { // expected-note {{here}}
    constexpr ~B() {} // expected-error {{destructor cannot be declared constexpr because base class 'subobject::A' does not have a constexpr destructor}}
  };
  struct C {
    A a; // expected-note {{here}}
    constexpr ~C() {} // expected-error {{destructor cannot be declared constexpr because data member 'a' does not have a constexpr destructor}}
  };
  struct D : A {
    A a;
    constexpr ~D() = delete;
  };
}
