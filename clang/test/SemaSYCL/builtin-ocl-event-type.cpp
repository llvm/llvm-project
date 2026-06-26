// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -verify -fsyntax-only %s
// expected-no-diagnostics

// Test that __ocl_event_t is available in SYCL device code

// A generic kernel launch function.
template<typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {
  t();
}

struct KernelName;

void test_ocl_event_t() {
  kernel_single_task<KernelName>([]() {
    // Test that __ocl_event_t is defined and can be used
    __ocl_event_t evt;
    (void)evt;
  });
}
