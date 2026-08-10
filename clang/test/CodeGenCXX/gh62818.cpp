// RUN: %clang_cc1 -std=c++17 -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s

void doSomething();

struct A {
  A() {};
  ~A() noexcept {
    doSomething();
  }

  A & operator=(A a) & noexcept {
    return *this;
  }
};

template<typename T>
struct B {
  void test() {a = {};}
  // CHECK: define linkonce_odr void @_ZN1BIiE4testEv
  // CHECK: call void @_ZN1AC1Ev(ptr noundef nonnull align 1 dereferenceable(1)
  // CHECK: [[CALL:%.*]] = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNR1AaSES_
  // CHECK: call void @_ZN1AD2Ev(ptr noundef nonnull align 1 dereferenceable(1)

  A a;
};

void client(B<int> &f) {f.test();}
