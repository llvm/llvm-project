// RUN: %clang --sycl -Xclang -fsycl-int-header=%t.h %s -c -o %T/kernel.spv
// RUN: FileCheck -input-file=%t.h %s
//
// CHECK: class first_kernel;
// CHECK-NEXT: template <typename T> class second_kernel;
// CHECK-NEXT: struct X;
// CHECK-NEXT: template <typename T> struct point ;
// CHECK-NEXT: template <int a, typename T1, typename T2> class third_kernel;
//
// CHECK: #include <CL/sycl/detail/kernel_desc.hpp>
//
// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "first_kernel",
// CHECK-NEXT:   "second_namespace::second_kernel<char>",
// CHECK-NEXT:   "third_kernel<1, int,  point< X> >"
// CHECK-NEXT: };
//
// CHECK: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   //--- first_kernel
// CHECK-NEXT:   { kernel_param_kind_t::kind_scalar, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 2014, 4 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 2016, 5 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- second_namespace::second_kernel<char>
// CHECK-NEXT:   { kernel_param_kind_t::kind_scalar, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 2016, 4 },
// CHECK-EMPTY:
// CHECK-NEXT:   //--- third_kernel<1, int,  point< X> >
// CHECK-NEXT:   { kernel_param_kind_t::kind_scalar, 4, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_accessor, 2016, 4 },
// CHECK-EMPTY:
// CHECK-NEXT: };
//
// CHECK: template <class KernelNameType> struct KernelInfo;
// CHECK: template <> struct KernelInfo<class first_kernel> {
// CHECK: template <> struct KernelInfo<class second_namespace::second_kernel<char>> {
// CHECK: template <> struct KernelInfo<class third_kernel<1, int, struct point<struct X> >> {

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
template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:
  void use(void) const {}
};
} // namespace sycl
} // namespace cl

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}
struct x {};
template <typename T>
struct point {};
namespace second_namespace {
template <typename T>
class second_kernel;
}

template <int a, typename T1, typename T2>
class third_kernel;

int main() {

  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc1;
  cl::sycl::accessor<float, 2, cl::sycl::access::mode::write,
                     cl::sycl::access::target::local,
                     cl::sycl::access::placeholder::true_t>
      acc2;
  int i = 13;
  // TODO: Uncomemnt when structures in kernel arguments are correctly processed
  //       by SYCL compiler
  /*  struct {
    char c;
    int i;
  } test_s;
  test_s.c = 14;*/
  kernel_single_task<class first_kernel>([=]() {
    if (i == 13 /*&& test_s.c == 14*/) {

      acc1.use();
      acc2.use();
    }
  });

  kernel_single_task<class second_namespace::second_kernel<char>>([=]() {
    if (i == 13) {
      acc2.use();
    }
  });
  kernel_single_task<class third_kernel<1, int,point<struct X>>>([=]() {
    if (i == 13) {
      acc2.use();
    }
  });

  return 0;
}

