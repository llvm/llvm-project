// RUN: %clang_cc1 -fsyntax-only %s -verify 
// expected-no-diagnostics

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
