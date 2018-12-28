// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -S -emit-llvm -x c++ %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {});
  return 0;
}

// CHECK: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CHECK: !spirv.Source = !{[[LANG:![0-9]+]]}
// CHECK: [[SPIR]] = !{i32 1, i32 2}
// CHECK: [[LANG]] = !{i32 4, i32 100000}
