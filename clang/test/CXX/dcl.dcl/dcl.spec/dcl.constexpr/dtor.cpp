// RUN: %clang_cc1 -std=c++2a -verify=expected,cxx2a %s
// RUN: %clang_cc1 -std=c++23 -verify=expected %s

// p3: if the function is a constructor or destructor, its class shall not have
// any virtual base classes;
namespace vbase {
  struct A {};
  struct B : virtual A { // expected-note {{virtual}}
    constexpr ~B() {} // expected-error {{constexpr member function not allowed in struct with virtual base class}}
  };
}

namespace contents {
  struct A {
    constexpr ~A() {
      return;
      goto x; // cxx2a-warning {{use of this statement in a constexpr function is a C++23 extension}}
      x: ;
    }
  };
  struct B {
    constexpr ~B() {
    x:; // cxx2a-warning {{use of this statement in a constexpr function is a C++23 extension}}
    }
  };
  struct Nonlit { // cxx2a-note {{'Nonlit' is not literal because}}
    Nonlit();
  };
  struct C {
    constexpr ~C() {
      return;
      Nonlit nl; // cxx2a-error {{variable of non-literal type 'Nonlit' cannot be defined in a constexpr function before C++23}}
    }
  };
  struct D {
    constexpr ~D() {
      return;
      static int a; // cxx2a-warning {{definition of a static variable in a constexpr function is a C++23 extension}}
    }
  };
  struct E {
    constexpr ~E() {
      return;
      thread_local int e; // cxx2a-warning {{definition of a thread_local variable in a constexpr function is a C++23 extension}}
    }
  };
  struct F {
    constexpr ~F() {
      return;
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
  struct B : A { // cxx2a-note {{here}}
    constexpr ~B() {} // cxx2a-error {{destructor cannot be declared constexpr because base class 'A' does not have a constexpr destructor}}
  };
  struct C {
    A a; // cxx2a-note {{here}}
    constexpr ~C() {} // cxx2a-error {{destructor cannot be declared constexpr because data member 'a' does not have a constexpr destructor}}
  };
  struct D : A {
    A a;
    constexpr ~D() = delete;
  };
}
