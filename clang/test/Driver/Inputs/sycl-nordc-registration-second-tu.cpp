// Second translation unit for sycl-nordc-registration.cpp. It contributes a
// kernel of its own so that a non-RDC build has to finalize and register two
// independent device binaries.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

struct second_tu_kernel_name;
struct second_tu_kernel {
  void operator()() const {}
};

[[clang::sycl_kernel_entry_point(second_tu_kernel_name)]]
void launch_second_tu_kernel(second_tu_kernel KernelFunc) {
  KernelFunc();
}

void call_second_tu() { launch_second_tu_kernel(second_tu_kernel{}); }
