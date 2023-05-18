// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wundefined-bool-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-bool-conversion -Wundefined-bool-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wbool-conversion %s

void test1(int &x) {
  if (x == 1) { }
  if (&x) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

  if (!&x) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
}

class test2 {
  test2() : x(y) {}

  void foo() {
    if (this) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}

    if (!this) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}
  }

  void bar() {
    if (x == 1) { }
    if (&x) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    if (!&x) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
  }

  int &x;
  int y;
};

namespace function_return_reference {
  int& get_int();
  // expected-note@-1 3{{'get_int' returns a reference}}
  class B {
  public:
    static int &stat();
    // expected-note@-1 3{{'stat' returns a reference}}
    int &get();
    // expected-note@-1 6{{'get' returns a reference}}
  };

  void test() {
    if (&get_int()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&(get_int())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&get_int()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    if (&B::stat()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&(B::stat())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&B::stat()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    B b;
    if (&b.get()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&(b.get())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&b.get()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    B* b_ptr = &b;
    if (&b_ptr->get()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&(b_ptr->get())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&b_ptr->get()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    int& (B::*m_ptr)() = &B::get;
    if (&(b.*m_ptr)()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&((b.*m_ptr)())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&(b.*m_ptr)()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    int& (*f_ptr)() = &get_int;
    if (&(*f_ptr)()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (&((*f_ptr)())) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
    if (!&(*f_ptr)()) {}
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
  }
}

namespace macros {
  #define assert(x) if (x) {}
  #define zero_on_null(x) ((x) ? *(x) : 0)

  void test(int &x) {
    // TODO: warn on assert(&x) but not on zero_on_null(&x)
    zero_on_null(&x);
    assert(zero_on_null(&x));
    assert(&x);

    assert(&x && "Expecting valid reference");
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
  }

  class S {
    void test() {
      assert(this);

      assert(this && "Expecting invalid reference");
      // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}
    }
  };
}
