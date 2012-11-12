// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -verify -o - |FileCheck %s

class x {
public: int operator=(int);
};
void a() {
  x a;
  a = 1u;
}

void f(int i, int j) {
  // CHECK: load i32
  // CHECK: load i32
  // CHECK: add nsw i32
  // CHECK: store i32
  // CHECK: store i32 17, i32
  // CHECK: ret
  (i += j) = 17;
}

// Taken from g++.old-deja/g++.jason/net.C
namespace test1 {
  template <class T> void fn (T t) { }
  template <class T> struct A {
    void (*p)(T);
    A() { p = fn; }
  };

  A<int> a;
}
