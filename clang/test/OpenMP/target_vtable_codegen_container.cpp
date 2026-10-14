// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52 -stdlib=libc++
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 -stdlib=libc++ | FileCheck %s
// expected-no-diagnostics

// CHECK-DAG: @_ZTV7Derived
// CHECK-DAG: @_ZTV4Base
template <typename T>
class Container {
private:
T value;
public:
Container() : value() {}
Container(T val) : value(val) {}

T getValue() const { return value; }

void setValue(T val) { value = val; }
};

class Base {
public:
    virtual void foo() {}
};
class Derived : public Base {};

class Test {
public:
    Container<Derived> v;
};

int main() {
  Test test;
  Derived d;
  test.v.setValue(d);

// Make sure we emit VTable for type indirectly (template specialized type)
#pragma omp target map(test)
  {
      test.v.getValue().foo();
  }
  return 0;
}
