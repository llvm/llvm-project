// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -Wno-openmp-target-exception -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -Wno-openmp-target-exception -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | \
// RUN: FileCheck -implicit-check-not='{{invoke|throw|cxa}}' %s
void foo() {
  // CHECK: call addrspace(9) void @llvm.trap()
  // CHECK-NEXT: call spir_func addrspace(9) void @__kmpc_target_deinit()
  #pragma omp target
  throw "bad";
}
