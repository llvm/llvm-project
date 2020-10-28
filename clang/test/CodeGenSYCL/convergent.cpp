// RUN: %clang_cc1 -fsycl -fsycl-is-device -emit-llvm -disable-llvm-passes \
// RUN:  -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | \
// RUN:   FileCheck %s

// CHECK-DAG: Function Attrs:
// CHECK-DAG-SAME: convergent
// CHECK-DAG-NEXT: define void @_Z3foov
void foo() {
  int a = 1;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { foo(); });
  return 0;
}
