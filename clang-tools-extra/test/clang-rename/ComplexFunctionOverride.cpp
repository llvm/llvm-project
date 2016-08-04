// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=307 -new-name=bar %t.cpp -i -- -std=c++11
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

struct A {
  virtual void foo();   // CHECK: virtual void bar();
};

struct B : A {
  void foo() override;  // CHECK: void bar() override;
};

struct C : B {
  void foo() override;  // CHECK: void bar() override;
};

struct D : B {
  void foo() override;  // CHECK: void bar() override;
};

struct E : D {
  void foo() override;  // CHECK: void bar() override;
};
