// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64 -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/Inputs/include -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

// Ensure we make `_ZdlPv`, aka. `operator delete(void*)` available without the need to `include <new>`.

// CHECK: define {{.*}}_ZdlPv

#ifndef HEADER
#define HEADER

class Base {
  public:
    virtual ~Base() = default;
};

class Derived : public Base {
  public:
    #pragma omp declare target
    Derived();
    #pragma omp end declare target
};

Derived::Derived() { }

int main(void) {
  #pragma omp target
  {
  }
  return 0;
}
#endif
