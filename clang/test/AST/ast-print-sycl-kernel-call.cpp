// RUN: %clang_cc1 -fsycl-is-host -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -ast-print %s -o - | FileCheck %s

struct sycl_kernel_launcher {
  template<typename KernelName, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) {}

  template<typename KernelName, typename KernelType>
  void kernel_entry_point(KernelType kernel) {
    kernel();
  }
// CHECK:      template <typename KernelName, typename KernelType> void kernel_entry_point(KernelType kernel) {
// CHECK-NEXT:     kernel();
// CHECK-NEXT: }
// CHECK:      template<> void kernel_entry_point<KN, (lambda at {{.*}})>((lambda at {{.*}}) kernel) {
// CHECK-NEXT:     kernel();
// CHECK-NEXT: }
};

void f(sycl_kernel_launcher skl) {
  skl.kernel_entry_point<struct KN>([]{});
}
