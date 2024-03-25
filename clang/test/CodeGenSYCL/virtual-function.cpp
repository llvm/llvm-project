// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Test verifying that RTTI information is not emitted
// during SYCL device compilation.

// CHECK-NOT: @_ZTI6Struct
 
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

struct Struct {
  virtual void foo() {}
  void bar() {}

};

int main() {
  kernel_single_task<class kernel_function>([]() {
                                            Struct S;
                                            S.bar(); });
  return 0;
}
