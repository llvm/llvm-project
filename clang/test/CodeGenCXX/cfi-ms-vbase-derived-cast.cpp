// RUN: %clang_cc1 -flto -flto-unit -emit-llvm -o - -triple=x86_64-pc-win32 %s -fsanitize=cfi-derived-cast -fsanitize-trap=cfi-derived-cast | FileCheck %s

struct foo {
  virtual ~foo() {}
  virtual void f() = 0;
};

template <typename T>
struct bar : virtual public foo {
  void f() {
    // CHECK: define{{.*}}@"\01?f@?$bar@Ubaz@@@@UEAAXXZ"
    // Load "this", vbtable, vbase offset and vtable.
    // CHECK: load
    // CHECK: load
    // CHECK: load
    // CHECK: load
    // CHECK: @llvm.type.test{{.*}}!"?AUfoo@@"
    static_cast<T&>(*this);
  }
};

struct baz : public bar<baz> {
  virtual ~baz() {}
};

int main() {
  baz *z = new baz;
  z->f();
}
