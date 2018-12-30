// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -S -emit-llvm %s -o - | FileCheck %s

namespace cl {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class placeholder { false_t,
                         true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space
};
} // namespace access

template <int dim>
struct range {
};

template <int dim>
struct _ImplT {
    range<dim> Range;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:

  void __set_pointer(__global dataT *Ptr) { }
  void __set_range(range<dimensions> Range) {
    __impl.Range = Range;
  }
  void use(void) const {}
  void use(void*) const {}
  _ImplT<dimensions> __impl;
};

} // namespace sycl
} // namespace cl

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0EE13__set_pointerEPU3AS1i(%"class.cl::sycl::accessor"* %1, i32 addrspace(1)* %2)
// CHECK: call spir_func void @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0EE11__set_rangeENS0_5rangeILi1EEE(%"class.cl::sycl::accessor"* %3, %"struct.cl::sycl::range"* byval align 1 %agg.tmp)
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
