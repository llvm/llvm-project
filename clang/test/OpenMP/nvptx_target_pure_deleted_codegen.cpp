// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fvisibility protected -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fvisibility protected -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fno-rtti | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fvisibility protected -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fno-rtti | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-NOT: class_type_info
// CHECK-DAG: @_ZTV7Derived = linkonce_odr protected unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%class.Derived*)* @_ZN7Derived3fooEv to i8*)] }
// CHECK-DAG: @_ZTV4Base = linkonce_odr protected unnamed_addr constant { [3 x i8*] } zeroinitializer
// CHECK-NOT: class_type_info
class Base {
  public:
  virtual void foo() = 0;
};

class Derived : public Base {
public:
  void foo() override {}
  void bar() = delete;
};

void target() {
#pragma omp target
  {
  Derived D;
  D.foo();
  }
}

#endif
