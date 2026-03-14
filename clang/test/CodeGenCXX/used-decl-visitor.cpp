// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o %t %s

// Make sure there is no assertion due to UsedDeclVisitor.

struct A {
  int a;
};

static A a;

struct B {
  B(int b = a.a) {}
};


void foo() {
  B();
}
