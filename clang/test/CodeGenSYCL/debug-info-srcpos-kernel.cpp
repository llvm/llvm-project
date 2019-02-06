// RUN: %clang --sycl %s -S -emit-llvm -g -o - | FileCheck %s
//
// Verify the SYCL kernel routine is marked artificial.
//
// Since it has no source correlation of its own, the SYCL kernel needs to be
// marked artificial or it will inherit source correlation from the surrounding
// code.
//

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  int value = 0;
  int* captured = &value;
  kernel_single_task<class kernel_function>([=]() {
      *captured = 1;
      });
  return 0;
}

// CHECK: define{{.*}} spir_kernel {{.*}}void @_ZTSZ4mainE15kernel_function(i32*{{.*}}){{.*}} !dbg [[KERNEL:![0-9]+]] {{.*}}{
// CHECK: [[FILE:![0-9]+]] = !DIFile(filename: "{{.*}}debug-info-srcpos-kernel.cpp"{{.*}})
// CHECK: [[KERNEL]] = {{.*}}!DISubprogram(name: "_ZTSZ4mainE15kernel_function"
// CHECK-SAME: scope: [[FILE]]
// CHECK-SAME: file: [[FILE]]
// CHECK-SAME: flags: DIFlagArtificial | DIFlagPrototyped
