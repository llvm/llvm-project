// RUN: %clang_cc1 -fsyntax-only -verify %s

// Various tests for -fno-exceptions

typedef __SIZE_TYPE__ size_t;

namespace test0 {
  class Foo {
  public:
    void* operator new(size_t x);
  private:
    void operator delete(void *x);
  };

  void test() {
    // Under -fexceptions, this does access control for the associated
    // 'operator delete'.
    (void) new Foo();
  }
}

namespace test1 {
void f() {
  throw; // expected-error {{cannot use 'throw' with exceptions disabled}}
}

void g() {
  try { // expected-error {{cannot use 'try' with exceptions disabled}}
    f();
  } catch (...) {
  }
}

}
