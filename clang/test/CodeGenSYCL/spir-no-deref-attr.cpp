// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm-bc -x c++ %s -o %t.bc
// RUN: llvm-spirv -spirv-no-deref-attr %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
struct foo {
  static void bar(int &val) {
    val = 1;
  }
};
int main() {
  kernel_single_task<class fake_kernel>([]() { int var; foo::bar(var); });
  return 0;
}

// CHECK: 3 Name [[os_ID:[0-9]+]] "val"
// CHECK-NOT: Decorate {{[0-9]+}} MaxByteOffset
