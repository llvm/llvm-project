// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s
// expected-no-diagnostics

namespace {

// Make sure both host and device compilation emit vtable for Dervied
// CHECK-DAG: @_ZTVN12_GLOBAL__N_17DerivedE
// CHECK-DAG: @_ZN12_GLOBAL__N_17DerivedD1Ev
// CHECK-DAG: @_ZN12_GLOBAL__N_17DerivedD0Ev
// CHECK-DAG: @_ZN12_GLOBAL__N_17Derived5BaseAEi
// CHECK-DAG: @_ZN12_GLOBAL__N_17Derived8DerivedBEv
class Base {
public:
  virtual ~Base() = default;
  virtual void BaseA(int a) { }
};

class Derived : public Base {
public:
  ~Derived() override = default;
  void BaseA(int a) override { x = a; }
  virtual void DerivedB() { }
private:
  int x;
};

};

int main() {

  Derived d;
  Base& c = d;
  int a = 50;
#pragma omp target data map (to: d, a)
  {
 #pragma omp target
     {
       c.BaseA(a);
     }
  }
  return 0;
}
