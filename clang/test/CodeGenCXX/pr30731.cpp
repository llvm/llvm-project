// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -flto -std=c++11 -o - %s | FileCheck %s

struct A {
  virtual ~A();
};

struct B {};

struct C {
  virtual void f();
};

struct S : A, virtual B, C {
  void f() override;
};

void f(S* s) { s->f(); }

// CHECK-LABEL: define void @"\01?f@@YAXPAUS@@@Z"
// CHECK: call
// CHECK: ret void
