// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin9 -o - %s | FileCheck %s

struct C {
  void f();
  void g(int, ...);
};

// CHECK-LABEL: define{{.*}} void @_ZN1C1fEv
void C::f() {
}

// CHECK-LABEL: define{{.*}} void @_Z5test1v
void test1() {
  C c;

  // CHECK: call void @_ZN1C1fEv
  c.f();

  // CHECK: call void (%struct.C*, i32, ...) @_ZN1C1gEiz
  c.g(1, 2, 3);
}


struct S {
  inline S() { }
  inline ~S() { }

  void f_inline1() { }
  inline void f_inline2() { }

  static void g() { }
  static void f();

  virtual void v() {}
};

// CHECK-LABEL: define{{.*}} void @_ZN1S1fEv
void S::f() {
}

void test2() {
  S s;

  s.f_inline1();
  s.f_inline2();

  S::g();
}

// S::S()
// CHECK: define linkonce_odr void @_ZN1SC1Ev{{.*}} unnamed_addr

// S::f_inline1()
// CHECK-LABEL: define linkonce_odr void @_ZN1S9f_inline1Ev

// S::f_inline2()
// CHECK-LABEL: define linkonce_odr void @_ZN1S9f_inline2Ev

// S::g()
// CHECK-LABEL: define linkonce_odr void @_ZN1S1gEv

// S::~S()
// CHECK: define linkonce_odr void @_ZN1SD1Ev{{.*}} unnamed_addr

struct T {
  T operator+(const T&);
};

// CHECK-LABEL: define{{.*}} void @_Z5test3v
void test3() {
  T t1, t2;

  // CHECK: call void @_ZN1TplERKS_
  T result = t1 + t2;
}

// S::S()
// CHECK: define linkonce_odr void @_ZN1SC2Ev{{.*}} unnamed_addr

// S::v()
// CHECK: define linkonce_odr void @_ZN1S1vEv{{.*}}unnamed_addr

// S::~S()
// CHECK: define linkonce_odr void @_ZN1SD2Ev{{.*}} unnamed_addr
