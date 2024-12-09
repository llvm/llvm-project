// REQUIRES: asserts

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -disable-llvm-passes -fopenmp-targets=amdgcn-amd-amdhsa -x c++ -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -target-cpu gfx906 -fopenmp -nogpulib -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -x c++ %s

// Don't crash with assertions build.
int MyGlobVar;
#pragma omp threadprivate(MyGlobVar)
int main() {
  MyGlobVar = 1;
}
