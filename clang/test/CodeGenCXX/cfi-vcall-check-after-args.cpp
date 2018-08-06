// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility hidden -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck %s

struct A {
  virtual void f(int);
};

int g();
void f(A *a) {
  // CHECK: call i32 @_Z1gv()
  // CHECK: call i1 @llvm.type.test
  a->f(g());
}
