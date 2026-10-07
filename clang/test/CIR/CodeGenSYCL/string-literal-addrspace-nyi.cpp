// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o /dev/null -verify

// SYCL device string literals belong in sycl_global, which CIR cannot
// represent yet.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
void use(const char *);

void test() {
  // expected-error@*:* {{ClangIR code gen Not Yet Implemented: SYCL global constant address space}}
  kernel_single_task<KN>([]() { use("hello"); });
}
