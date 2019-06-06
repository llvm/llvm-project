// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper arguments for
// different accessors targets.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  accessor<int, 1, access::mode::read_write,
           access::target::local>
      local_acc;
  accessor<int, 1, access::mode::read_write,
           access::target::global_buffer>
      global_acc;
  accessor<int, 1, access::mode::read_write,
           access::target::constant_buffer>
      constant_acc;
  kernel<class use_local>(
      [=]() {
        local_acc.use();
      });
  kernel<class use_global>(
      [=]() {
        global_acc.use();
      });
  kernel<class use_constant>(
      [=]() {
        constant_acc.use();
      });
}
// CHECK: {{.*}}use_local 'void (__local int *, range<1>, range<1>, id<1>)'
// CHECK: {{.*}}use_global 'void (__global int *, range<1>, range<1>, id<1>)'
// CHECK: {{.*}}use_constant 'void (__constant int *, range<1>, range<1>, id<1>)'
