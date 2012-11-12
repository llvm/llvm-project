// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Base {
  int data;
  int method();
};
int (Base::*data_ptr) = &Base::data;
int (Base::*method_ptr)() = &Base::method;

namespace test0 {
  struct Derived : Base {};
  void test() {
    int (Derived::*d) = data_ptr;
    int (Derived::*m)() = method_ptr;
  }
}

// Can't be inaccessible.
namespace test1 {
  struct Derived : private Base {}; // expected-note 2 {{declared private here}}
  void test() {
    int (Derived::*d) = data_ptr; // expected-error {{cannot cast private base class 'Base' to 'test1::Derived'}}
    int (Derived::*m)() = method_ptr; // expected-error {{cannot cast private base class 'Base' to 'test1::Derived'}}
  }
};

// Can't be ambiguous.
namespace test2 {
  struct A : Base {};
  struct B : Base {};
  struct Derived : A, B {};
  void test() {
    int (Derived::*d) = data_ptr; // expected-error {{ambiguous conversion from pointer to member of base class 'Base' to pointer to member of derived class 'test2::Derived':}}
    int (Derived::*m)() = method_ptr; // expected-error {{ambiguous conversion from pointer to member of base class 'Base' to pointer to member of derived class 'test2::Derived':}}
  }
}

// Can't be virtual.
namespace test3 {
  struct Derived : virtual Base {};
  void test() {
    int (Derived::*d) = data_ptr;  // expected-error {{conversion from pointer to member of class 'Base' to pointer to member of class 'test3::Derived' via virtual base 'Base' is not allowed}}
    int (Derived::*m)() = method_ptr; // expected-error {{conversion from pointer to member of class 'Base' to pointer to member of class 'test3::Derived' via virtual base 'Base' is not allowed}}
  }
}

// Can't be virtual even if there's a non-virtual path.
namespace test4 {
  struct A : Base {};
  struct Derived : Base, virtual A {};
  void test() {
    int (Derived::*d) = data_ptr; // expected-error {{ambiguous conversion from pointer to member of base class 'Base' to pointer to member of derived class 'test4::Derived':}}
    int (Derived::*m)() = method_ptr; // expected-error {{ambiguous conversion from pointer to member of base class 'Base' to pointer to member of derived class 'test4::Derived':}}
  }
}

// PR6254: don't get thrown off by a virtual base.
namespace test5 {
  struct A {};
  struct Derived : Base, virtual A {};
  void test() {
    int (Derived::*d) = data_ptr;
    int (Derived::*m)() = method_ptr;
  }
}
