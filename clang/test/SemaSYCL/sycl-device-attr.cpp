// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void foo() {
  kernel<class kernel_name>([]() {});
}

// CHECK: SYCLDeviceAttr
