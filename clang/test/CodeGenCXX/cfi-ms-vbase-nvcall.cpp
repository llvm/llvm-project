// RUN: %clang_cc1 -flto -flto-unit -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-nvcall -fsanitize-trap=cfi-nvcall | FileCheck %s

struct foo {
  virtual ~foo() {}
  virtual void f() = 0;
};

template <typename T>
struct bar : virtual public foo {
  void f() {}
};

struct baz : public bar<baz> {
  virtual ~baz() {}
  void g() {}
};

void f(baz *z) {
  // CHECK: define{{.*}}@"?f@@YAXPEAUbaz@@@Z"
  // Load z, vbtable, vbase offset and vtable.
  // CHECK: load
  // CHECK: load
  // CHECK: load
  // CHECK: load
  // CHECK: @llvm.type.test{{.*}}!"?AUfoo@@"
  z->g();
}
