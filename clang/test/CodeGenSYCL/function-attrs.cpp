// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -disable-llvm-passes \
// RUN:  -triple spir64 -fexceptions -emit-llvm %s -o - | FileCheck %s

int foo();

// CHECK: define dso_local spir_func void @_Z3barv() [[BAR:#[0-9]+]]
// CHECK: attributes [[BAR]] =
// CHECK-SAME: convergent
// CHECK-SAME: nounwind
void bar() {
  int a = foo();
}

int foo() {
  return 1;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { bar(); });
  return 0;
}
