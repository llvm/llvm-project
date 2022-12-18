// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa  -dwarf-version=2 -debugger-tuning=gdb -debug-info-kind=constructor -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -dwarf-version=4 -debugger-tuning=gdb -fcuda-is-device -debug-info-kind=constructor -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK

// expected-no-diagnostics

// CHECK: private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}};main;
int main (void)
{
  int res = 0;
#pragma omp target map(res)
#pragma omp parallel for reduction(+:res)
    for (int i = 0; i < 10; i++) {
      res += i;
    }

  return res;
}
