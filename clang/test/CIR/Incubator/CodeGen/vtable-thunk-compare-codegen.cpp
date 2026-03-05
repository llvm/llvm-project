// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test that CIR thunk generation matches original CodeGen behavior

class Base1 {
public:
  virtual void foo() {}
};

class Base2 {
public:
  virtual void bar() {}
};

class Derived : public Base1, public Base2 {
public:
  void bar() override {}
};

void test() {
  Derived d;
  Base2* b2 = &d;
  b2->bar();
}

// Check CIR thunk in vtable
// CIR: cir.global {{.*}}linkonce_odr @_ZTV7Derived = #cir.vtable<{{.*}}@_ZThn{{[0-9]+}}_N7Derived3barEv

// Check CIR thunk function
// CIR: cir.func {{.*}} @_ZThn{{[0-9]+}}_N7Derived3barEv
// CIR:   cir.ptr_stride
// CIR:   cir.call @_ZN7Derived3barEv

// Check LLVM thunk in vtable (from CIR)
// LLVM-DAG: @_ZTV7Derived = linkonce_odr constant {{.*}} @_ZThn{{[0-9]+}}_N7Derived3barEv

// Check LLVM thunk function (from CIR)
// LLVM-DAG: define linkonce_odr void @_ZThn{{[0-9]+}}_N7Derived3barEv

// Check original CodeGen LLVM output matches
// OGCG-DAG: @_ZTV7Derived = linkonce_odr unnamed_addr constant {{.*}} @_ZThn{{[0-9]+}}_N7Derived3barEv
// OGCG-DAG: define linkonce_odr void @_ZThn{{[0-9]+}}_N7Derived3barEv
