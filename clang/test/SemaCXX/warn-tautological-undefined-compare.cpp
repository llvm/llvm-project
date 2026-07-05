// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-undefined-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-tautological-compare -Wtautological-undefined-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-compare %s

void test1(int &x) {
  if (x == 1) { }
  if (&x == 0) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
  if (&x != 0) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
}

class test2 {
  test2() : x(y) {}

  void foo() {
    if (this == 0) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (this != 0) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to true}}
  }

  void bar() {
    if (x == 1) { }
    if (&x == 0) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&x != 0) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
  }

  int &x;
  int y;
};

namespace function_return_reference {
  int& get_int();
  // expected-note@-1 4{{'get_int' returns a reference}}
  class B {
  public:
    static int &stat();
    // expected-note@-1 4{{'stat' returns a reference}}
    int &get();
    // expected-note@-1 8{{'get' returns a reference}}
  };

  void test() {
    if (&get_int() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&(get_int()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&get_int() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&(get_int()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

    if (&B::stat() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&(B::stat()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&B::stat() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&(B::stat()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

    B b;
    if (&b.get() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&(b.get()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&b.get() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&(b.get()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

    B* b_ptr = &b;
    if (&b_ptr->get() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&(b_ptr->get()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&b_ptr->get() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&(b_ptr->get()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

    int& (B::*m_ptr)() = &B::get;
    if (&(b.*m_ptr)() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&((b.*m_ptr)()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&(b.*m_ptr)() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&((b.*m_ptr)()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}

    int& (*f_ptr)() = &get_int;
    if (&(*f_ptr)() == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    if (&((*f_ptr)()) == 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}

    if (&(*f_ptr)() != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    if (&((*f_ptr)()) != 0) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
  }
}

namespace macros {
  #define assert(x) if (x) {}

  void test(int &x) {
    assert(&x != 0);
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    assert(&x == 0);
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    assert(&x != 0 && "Expecting valid reference");
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to true}}
    assert(&x == 0 && "Expecting invalid reference");
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; comparison may be assumed to always evaluate to false}}
  }

  class S {
    void test() {
      assert(this != 0);
      // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to true}}
      assert(this == 0);
      // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to false}}
      assert(this != 0 && "Expecting valid reference");
      // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to true}}
      assert(this == 0 && "Expecting invalid reference");
      // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; comparison may be assumed to always evaluate to false}}
    }
  };
}
