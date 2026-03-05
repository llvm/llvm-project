// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: FileCheck --input-file=%t.ogcg.ll %s

// XFAIL: *

// Test thunk generation for virtual destructors in multiple inheritance

class Base1 {
public:
  virtual ~Base1() {}
  int x;
};

class Base2 {
public:
  virtual ~Base2() {}
  int y;
};

class Derived : public Base1, public Base2 {
public:
  ~Derived() override {}
};

void test() {
  Base2* b2 = new Derived();
  delete b2;  // Uses destructor thunk
}

// CHECK-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD1Ev
//       CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 -16
//       CHECK: call void @_ZN7DerivedD1Ev

// CHECK-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD0Ev
//       CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 -16
//       CHECK: call void @_ZN7DerivedD0Ev
