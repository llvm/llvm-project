// RUN: %clang_cc1 -DCK1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=52
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -Wno-openmp-mapping -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode  -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited -fopenmp-version=52 | FileCheck %s --check-prefix=CK1
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef CK1

#pragma omp begin declare target

class Base {
public:
  virtual int foo() { return 1; }
  virtual int bar() { return 2; }
};

class Derived : public Base {
public:
  virtual int foo() { return 3; }
  virtual int bar() { return 4; }
};

#pragma omp end declare target

int main() {
  Base base;
  Derived derived;
  {
#pragma omp target data map(base, derived)
    {
      Base *pointer1 = &base;
      Base *pointer2 = &derived;

#pragma omp target
      {
        // CK1-DAG:  call ptr @__llvm_omp_indirect_call_lookup(ptr %vtable{{[0-9]*}})
        // CK1-DAG:  call ptr @__llvm_omp_indirect_call_lookup(ptr %vtable{{[0-9]*}})
        // CK1-DAG:  call ptr @__llvm_omp_indirect_call_lookup(ptr %vtable{{[0-9]*}})
        // CK1-DAG:  call ptr @__llvm_omp_indirect_call_lookup(ptr %vtable{{[0-9]*}})
        int result1 = pointer1->foo();
        int result2 = pointer1->bar();
        int result3 = pointer2->foo();
        int result4 = pointer2->bar();
      }
    }
  }
  return 0;
}

#endif
#endif
