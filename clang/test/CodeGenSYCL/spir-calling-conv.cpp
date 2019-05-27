// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {

  // CHECK: define spir_kernel void @_ZTSZ4mainE15kernel_function()

  // CHECK: call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}.anon"* %0)

  // CHECK: define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}anon"* %this)

  kernel_single_task<class kernel_function>([]() {});
  return 0;
}
