// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgpu9.00-amd-amdhsa -x hip -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -x hip -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ast-dump -ast-dump-filter=gh45327 %s | FileCheck %s --check-prefix=AST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-pch -o %t.cuda %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -include-pch %t.cuda -x cuda -ast-dump-all -ast-dump-filter=gh45327 /dev/null | FileCheck %s --check-prefix=AST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -x hip -emit-pch -o %t.hip %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -include-pch %t.hip -x hip -ast-dump-all -ast-dump-filter=gh45327 /dev/null | FileCheck %s --check-prefix=AST

#include "Inputs/cuda.h"

namespace gh45327 {
// The launch translation unit in the issue sees only this friend declaration;
// init is defined in another translation unit.
class MyClass {
public:
  friend __global__ void init(MyClass *simulation_map);
};

// HOST-LABEL: define{{.*}} @_ZN7gh453274testEv(
// HOST: call void @_ZN7gh45327{{.*}}initEPNS_7MyClassE(
// AST-LABEL: FunctionDecl {{.*}} test 'void ()'
// AST: CUDAKernelCallExpr {{.*}} 'void' adl
void test() {
  MyClass obj;
  init<<<1, 1>>>(&obj);
}
} // namespace gh45327

namespace associated {
struct Arg {};

// DEVICE-LABEL: define{{.*}} @_ZN10associated6kernelENS_3ArgE(
__global__ void kernel(Arg) {}

// DEVICE-LABEL: define{{.*}} @_ZN10associated15template_kernelIiEEvNS_3ArgET_(
template <class T> __global__ void template_kernel(Arg, T) {}
} // namespace associated

// HOST-LABEL: define{{.*}} @_Z6launchv(
// HOST: call void @_ZN10associated{{.*}}kernelENS_3ArgE(
// HOST: call void @_ZN10associated{{.*}}template_kernelIiEEvNS_3ArgET_(
void launch() {
  associated::Arg arg;
  kernel<<<1, 1>>>(arg);
  // An explicit template argument list takes the C++ id-expression path.
  template_kernel<int><<<1, 1>>>(arg, 0);
}

// Lookup of the kernel is deferred until this function is instantiated.
template <class T> void dependent_launch(T arg) {
  kernel<<<1, 1>>>(arg);
}
template void dependent_launch(associated::Arg);
