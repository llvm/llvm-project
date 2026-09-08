// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fherbceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fherbceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fherbceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Herbception special functions: constructors, copy constructors, operator
// overloading, and destructors with throws.

namespace std {
struct error { void *d; __SIZE_TYPE__ c; };
}

// Constructor with throws
struct Foo {
  int x;
  Foo(int x) throws : x(x) {}
  Foo(const Foo& other) throws : x(other.x) {}
  ~Foo() noexcept = default;
};

// CIR: @_ZN3FooC1
// CIR: @_ZN3FooC2

// LLVM: define{{.*}} @_ZN3FooC
// OGCG: define{{.*}} @_ZN3FooC

int main() {
  Foo f(10);
  Foo f2 = f;
  return 0;
}
