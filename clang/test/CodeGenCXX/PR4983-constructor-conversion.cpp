// RUN: %clang_cc1 -emit-llvm-only %s

struct A {
  A(const char *s){}
};

struct B {
  A a;
  
  B() : a("test") { }
};

void f() {
    A a("test");
}

