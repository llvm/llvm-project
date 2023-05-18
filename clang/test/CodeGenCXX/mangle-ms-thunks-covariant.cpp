// RUN: %clang_cc1 -fno-rtti-data -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=x86_64-pc-win32 -fms-compatibility-version=19.00 | FileCheck %s --check-prefix=CHECK

namespace t1 {
struct A {
public:
  virtual ~A();
  virtual A *f();
};
struct B {
public:
  virtual ~B();

private:
  virtual B *f();
};
struct C : A, B {
  virtual ~C();

protected:
  virtual C *f();
};
C c;
}
// Main external C::f impl:
// CHECK-DAG: "?f@C@t1@@MEAAPEAU12@XZ"
// New slot in C's vftable for B, returns C* directly:
// CHECK-DAG: "?f@C@t1@@O7EAAPEAU12@XZ"
// Return-adjusting thunk in C's vftable for B:
// CHECK-DAG: "?f@C@t1@@W7EAAPEAUB@2@XZ"
