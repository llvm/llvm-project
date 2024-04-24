// RUN: %clang_cc1 -fsyntax-only %s -verify 

namespace rdar10544564 {
  // Check that we don't attempt to use an overloaded operator& when
  // naming a pointer-to-member.
  struct X {
    void** operator & ();
  };

  struct Y
  {
  public:
    X member;
    X memfunc1();
    X memfunc2();
    X memfunc2(int);

    void test() {
      X Y::*data_mem_ptr = &Y::member;
      X (Y::*func_mem_ptr1)() = &Y::memfunc1;
      X (Y::*func_mem_ptr2)() = &Y::memfunc2;
    }
  };
  
  X Y::*data_mem_ptr = &Y::member;
  X (Y::*func_mem_ptr1)() = &Y::memfunc1;
  X (Y::*func_mem_ptr2)() = &Y::memfunc2;
}

namespace GH40906 {
  struct A {
    int val;
    void func() {}
  };

  void test() {
    decltype(&(A::val)) ptr1; // expected-error {{invalid use of non-static data member 'val'}}
    int A::* ptr2 = &(A::val); // expected-error {{invalid use of non-static data member 'val'}}

    // FIXME: Error messages in these cases are less than clear, we can do
    // better.
    int size = sizeof(&(A::func)); // expected-error {{call to non-static member function without an object argument}}
    void (A::* ptr3)() = &(A::func); // expected-error {{call to non-static member function without an object argument}}
  }
}
