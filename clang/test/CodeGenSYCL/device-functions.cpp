// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s

template <typename T>
T bar(T arg);

void foo() {
  int a = 1 + 1 + bar(1);
}

template <typename T>
T bar(T arg) {
  return arg;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { foo(); });
  return 0;
}
// CHECK: define spir_kernel void @_ZTSZ4mainE11fake_kernel()
// CHECK: define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}.anon"* %this)
// CHECK: define spir_func void @_Z3foov()
// CHECK: define linkonce_odr spir_func i32 @_Z3barIiET_S0_(i32 %arg)
