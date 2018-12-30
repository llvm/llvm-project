// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

#include <sycl.hpp>

namespace foo {
namespace cl {
namespace sycl {
class accessor {
public:
  int field;
};
} // namespace sycl
} // namespace cl
} // namespace foo

class accessor {
public:
  int field;
};

typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>
    MyAccessorTD;

using MyAccessorA = cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::global_buffer>;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  foo::cl::sycl::accessor acc = {1};
  accessor acc1 = {1};

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorB;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorC;
    kernel<class fake_accessors>(
        [=]() {
          accessorA.use((void*)(acc.field + acc1.field));
        });
    kernel<class accessor_typedef>(
        [=]() {
          accessorB.use((void*)(acc.field + acc1.field));
        });
    kernel<class accessor_alias>(
        [=]() {
          accessorC.use((void*)(acc.field + acc1.field));
        });
  return 0;
}
// CHECK: fake_accessors 'void (__global int *, range<1>, id<1>, foo::cl::sycl::accessor, accessor)
// CHECK: accessor_typedef 'void (__global int *, range<1>, id<1>, foo::cl::sycl::accessor, accessor)
// CHECK: accessor_alias 'void (__global int *, range<1>, id<1>, foo::cl::sycl::accessor, accessor)
