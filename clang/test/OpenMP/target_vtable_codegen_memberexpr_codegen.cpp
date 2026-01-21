// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s
// expected-no-diagnostics


// CHECK-DAG: $_ZN4Base5BaseAEi = comdat any
// CHECK-DAG: $_ZN7Derived5BaseAEi = comdat any
// CHECK-DAG: $_ZN7Derived8DerivedBEv = comdat any
// CHECK-DAG: $_ZN4BaseD1Ev = comdat any
// CHECK-DAG: $_ZN4BaseD0Ev = comdat any
// CHECK-DAG: $_ZN7DerivedD1Ev = comdat any
// CHECK-DAG: $_ZN7DerivedD0Ev = comdat any
// CHECK-DAG: $_ZN4BaseD2Ev = comdat any
// CHECK-DAG: $_ZN7DerivedD2Ev = comdat any
// CHECK-DAG: $_ZTV4Base = comdat any
// CHECK-DAG: $_ZTV7Derived = comdat any
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

struct VirtualContainer {
  Base baseObj;
  Derived derivedObj;
  Base *basePtr;
};

int main() {
  VirtualContainer container;
  container.basePtr = &container.derivedObj;
  int a = 50;
#pragma omp target map(container.baseObj, container.derivedObj,                \
                           container.basePtr[ : 1])
  {
    container.baseObj.BaseA(a);
    container.derivedObj.BaseA(a);
    container.derivedObj.DerivedB();
    container.basePtr->BaseA(a);
  }
  return 0;
}
