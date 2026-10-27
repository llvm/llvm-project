// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s
// expected-no-diagnostics

// Make sure both host and device compilation emit vtable for Dervied
// CHECK-DAG: $_ZN7DerivedD1Ev = comdat any
// CHECK-DAG: $_ZN7DerivedD0Ev = comdat any
// CHECK-DAG: $_ZN7Derived5BaseAEi = comdat any
// CHECK-DAG: $_ZN7Derived8DerivedBEv = comdat any
// CHECK-DAG: $_ZN7DerivedD2Ev = comdat any
// CHECK-DAG: $_ZN4BaseD2Ev = comdat any
// CHECK-DAG: $_ZTV7Derived = comdat any
class Base {
public:

  virtual ~Base() = default;

  virtual void BaseA(int a) { }
};

// CHECK: @_ZTV7Derived = linkonce_odr unnamed_addr constant { [6 x ptr] }
class Derived : public Base {
public:

  ~Derived() override = default;

  void BaseA(int a) override { x = a; }

  virtual void DerivedB() { }
private:
  int x;
};

int main() {

  Derived d;
  Base& c = d;
  int a = 50;
  // Should emit vtable for Derived since d is added to map clause
#pragma omp target data map (to: d, a)
  {
 #pragma omp target map(d)
     {
       c.BaseA(a);
     }
  }
  return 0;
}
