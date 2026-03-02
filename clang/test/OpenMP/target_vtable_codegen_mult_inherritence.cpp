// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s
// expected-no-diagnostics

// CHECK-DAG: @_ZTV6Base_1
// CHECK-DAG: @_ZTV7Derived
// CHECK-DAG: @_ZTV6Base_2
#pragma omp begin declare target

class Base_1 {
public:
  virtual void foo() { }
  virtual void bar() { }
};

class Base_2 {
public:
  virtual void foo() { }
  virtual void bar() { }
};

class Derived : public Base_1, public Base_2 {
public:
  virtual void foo() override { }
  virtual void bar() override { }
};

#pragma omp end declare target

int main() {
  Base_1 base;
  Derived derived;

  // Make sure we emit vtable for parent class (Base_1 and Base_2)
#pragma omp target data map(derived)
    {
      Base_1 *p1 = &derived;

#pragma omp target
      {
        p1->foo();
        p1->bar();
      }
    }
  return 0;
}
