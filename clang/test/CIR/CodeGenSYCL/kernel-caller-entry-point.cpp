// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-cir -verify %s

// During device compilation, a SYCL kernel caller offload entry point is
// emitted in place of each sycl_kernel_entry_point attributed function. That
// lowering is not yet implemented in CIR, so it must be reported as a clean
// "Not Yet Implemented" diagnostic rather than crashing.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
// expected-error@+1 {{ClangIR code gen Not Yet Implemented: SYCL kernel caller offload entry point}}
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
struct K {
  void operator()() const {}
};

void test() { kernel_single_task<KN>(K{}); }
