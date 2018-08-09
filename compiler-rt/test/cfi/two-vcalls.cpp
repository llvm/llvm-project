// RUN: %clangxx_cfi_diag -o %t %s
// RUN: %run %t 2>&1 | FileCheck %s

// This test checks that we don't generate two type checks,
// if two virtual calls are in the same function.

// UNSUPPORTED: windows-msvc
// REQUIRES: cxxabi

// TODO(krasin): implement the optimization to not emit two type checks.
// XFAIL: *
#include <stdio.h>

class Base {
 public:
  virtual void Foo() {
    fprintf(stderr, "Base::Foo\n");
  }

  virtual void Bar() {
    fprintf(stderr, "Base::Bar\n");
  }
};

class Derived : public Base {
 public:
  void Foo() override {
    fprintf(stderr, "Derived::Foo\n");
  }

  void Bar() override {
    printf("Derived::Bar\n");
  }
};

__attribute__((noinline)) void print(Base* ptr) {
  ptr->Foo();
  // Corrupt the vtable pointer. We expect that the optimization will
  // check vtable before the first vcall then store it in a local
  // variable, and reuse it for the second vcall. With no optimization,
  // CFI will complain about the virtual table being corrupted.
  *reinterpret_cast<void**>(ptr) = 0;
  ptr->Bar();
}


int main() {
  Base b;
  Derived d;
  // CHECK: Base::Foo
  // CHECK: Base::Bar
  print(&b);

  // CHECK: Derived::Foo
  // CHECK-NOT: runtime error
  // CHECK: Derived::Bar
  print(&d);

  return 0;
}
