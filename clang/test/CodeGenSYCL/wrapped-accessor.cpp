// RUN: %clang -I %S/Inputs --sycl -Xclang -fsycl-int-header=%t.h %s -c -o %T/kernel.spv
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>

// CHECK: class wrapped_access;

// CHECK: namespace cl {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZ4mainE14wrapped_access"
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT: //--- _ZTSZ4mainE14wrapped_access
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 12, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 4062, 0 },
// CHECK-EMPTY:
// CHECK-NEXT: };

// CHECK: static constexpr
// CHECK-NEXT: const unsigned kernel_signature_start[] = {
// CHECK-NEXT:  0 // _ZTSZ4mainE14wrapped_access
// CHECK-NEXT: };

// CHECK: template <class KernelNameType> struct KernelInfo;

// CHECK: template <> struct KernelInfo<class wrapped_access> {

#include <sycl.hpp>

template <typename Acc>
struct AccWrapper { Acc accessor; };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;
  auto acc_wrapped = AccWrapper<decltype(acc)>{acc};
  kernel<class wrapped_access>(
      [=]() {
        acc_wrapped.accessor.use();
      });
}
