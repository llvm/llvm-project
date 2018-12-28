// RUN: %clang -S --sycl -Xclang -ast-dump %s | FileCheck %s
// XFAIL: *
#include <CL/sycl.hpp>

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

int main() {
  int data = 5;
  cl::sycl::queue deviceQueue;
  cl::sycl::buffer<int, 1> bufferA(&data, cl::sycl::range<1>(1));
  foo::cl::sycl::accessor acc = {1};
  accessor acc1 = {1};

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<cl::sycl::access::mode::read_write>(cgh);
    MyAccessorTD accessorB = bufferA.template get_access<cl::sycl::access::mode::read_write>(cgh);
    MyAccessorA accessorC = bufferA.template get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task<class fake_accessors>(
        [=]() {
          accessorA[0] = acc.field + acc1.field;
        });
    cgh.single_task<class accessor_typedef>(
        [=]() {
          accessorB[0] = acc.field + acc1.field;
        });
    cgh.single_task<class accessor_alias>(
        [=]() {
          accessorC[0] = acc.field + acc1.field;
        });
  });
  return 0;
}
// CHECK: fake_accessors 'void (__global int *__global, foo::cl::sycl::accessor, accessor)
// CHECK: accessor_typedef 'void (__global int *__global, foo::cl::sycl::accessor, accessor)
// CHECK: accessor_alias 'void (__global int *__global, foo::cl::sycl::accessor, accessor)
